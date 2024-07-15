use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use ngrammatic::{CorpusBuilder, Corpus, Pad};
use web_sys::console;
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use std::io::prelude::*;

const EMBEDDING_SIZE: usize = 1000;
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.1;
const K1: f32 = 1.2;
const B: f32 = 0.75;

fn compress_ngram_vector(vector: &[f32]) -> Vec<u8> {
  let mut delta_encoded: Vec<i32> = Vec::with_capacity(vector.len());
  let mut prev = 0.0;
  for &value in vector {
    let delta = (value * 1000.0) as i32 - (prev * 1000.0) as i32;
    delta_encoded.push(delta);
    prev = value;
  }

  let bytes: Vec<u8> = delta_encoded.iter().flat_map(|&x| x.to_le_bytes()).collect();

  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  encoder.write_all(&bytes).unwrap();
  encoder.finish().unwrap()
}

fn decompress_ngram_vector(compressed: &[u8]) -> Vec<f32> {
  let mut decoder = ZlibDecoder::new(compressed);
  let mut decompressed = Vec::new();
  decoder.read_to_end(&mut decompressed).unwrap();

  let mut delta_encoded: Vec<i32> = Vec::new();
  for chunk in decompressed.chunks_exact(4) {
    delta_encoded.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
  }

  let mut vector: Vec<f32> = Vec::with_capacity(delta_encoded.len());
  let mut prev = 0;
  for &delta in &delta_encoded {
    prev += delta;
    vector.push(prev as f32 / 1000.0);
  }

  vector
}


/// Represents a document in the index with its associated metadata and vector representation.
#[derive(Serialize, Deserialize, Clone)]
struct IndexedDocument {
  name: String,
  paths: HashSet<String>,
  word_frequencies: HashMap<String, u32>,
  ngram_vector: Vec<f32>,
}

impl IndexedDocument {
  fn new(name: &str, paths: &HashSet<String>, content: &str, ngram_vector: Vec<f32>) -> Self {
    let mut word_frequencies = HashMap::new();
    for word in content.split_whitespace() {
      *word_frequencies.entry(word.to_lowercase()).or_insert(0) += 1;
    }

    IndexedDocument {
      name: name.to_string(),
      paths: paths.clone(),
      word_frequencies,
      ngram_vector,
    }
  }
}

#[derive(Serialize, Deserialize)]
struct SerializableDocument {
  name: String,
  paths: Vec<String>,
  compressed_ngram_vector: Vec<u8>,
  word_frequencies: HashMap<String, u32>,
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export type IndexData = {
    [key: string]: {
        name: string;
        paths: string[];
        word_frequencies: { [word: string]: number };
        ngram_vector: Float32Array;
    }
};

export type SearchResult = {
    similarity: number;
    name: string;
    paths: string[];
};

export interface VectorizationSystemConstructor {
    new(): VectorizationSystem;
    new(index_data: IndexData): VectorizationSystem;
}
"#;

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(typescript_type = "IndexData")]
  pub type IndexDataJS;

  #[wasm_bindgen(typescript_type = "SearchResult[]")]
  pub type SearchResultArrayJS;
}

/// VectorizationSystem provides methods for text vectorization and similarity search.
#[wasm_bindgen]
pub struct VectorizationSystem {
  corpus: Corpus,
  documents: HashMap<String, IndexedDocument>,
  similarity_threshold: f32,
}

#[wasm_bindgen]
impl VectorizationSystem {
  /// Creates a new VectorizationSystem.
  #[wasm_bindgen(constructor)]
  pub fn new() -> Self {
    console_error_panic_hook::set_once();
    VectorizationSystem {
      corpus: CorpusBuilder::new()
        .arity(2)
        .pad_full(Pad::Auto)
        .finish(),
      documents: HashMap::new(),
      similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
    }
  }

  /// Adds or updates a document in the index.
  /// @param name - The name of the document.
  /// @param path - The path or identifier of the document.
  /// @param content - The content of the document.
  #[wasm_bindgen(js_name = addOrUpdateDocument)]
  pub fn add_or_update_document(&mut self, name: &str, path: &str, content: &str) {
    console::log_1(&format!("Adding document: {}", name).into());

    if content.len() < 10 {
      console::log_1(&format!("Document '{}' is too short, skipping", name).into());
      return;
    }

    let ngram_vector = self.calculate_vector(content);

    if ngram_vector.iter().any(|&x| x.is_nan()) || ngram_vector.iter().all(|&x| x == 0.0) {
      console::log_1(&format!("Warning: Invalid vector for document '{}'", name).into());
      return;
    }

    let mut paths = self.documents.get(name)
      .map(|doc| doc.paths.clone())
      .unwrap_or_else(HashSet::new);
    paths.insert(path.to_string());

    self.documents.insert(name.to_string(), IndexedDocument::new(name, &paths, content, ngram_vector));

    self.corpus.add_text(content);
    console::log_1(&format!("Document added: {}", name).into());
  }


  /// Adds a new path to an existing document.
  /// @param name - The name of the document.
  /// @param path - The new path to add.
  #[wasm_bindgen(js_name = addPath)]
  pub fn add_path(&mut self, name: &str, path: &str) -> Result<(), JsValue> {
    if let Some(doc) = self.documents.get_mut(name) {
      doc.paths.insert(path.to_string());
      Ok(())
    } else {
      Err(JsValue::from_str(&format!("Document '{}' not found", name)))
    }
  }

  /// Updates the content of an existing document.
  /// @param name - The name of the document.
  /// @param content - The new content of the document.
  #[wasm_bindgen(js_name = updateContent)]
  pub fn update_content(&mut self, name: &str, content: &str) -> Result<(), JsValue> {
    if !self.documents.contains_key(name) {
      return Err(JsValue::from_str(&format!("Document '{}' not found", name)));
    }

    let ngram_vector = self.calculate_vector(content);

    if ngram_vector.iter().any(|&x| x.is_nan()) || ngram_vector.iter().all(|&x| x == 0.0) {
      return Err(JsValue::from_str(&format!("Invalid vector for document '{}'", name)));
    }

    if let Some(doc) = self.documents.get_mut(name) {
      doc.ngram_vector = ngram_vector;
    }

    self.corpus.add_text(content);

    Ok(())
  }

  /// Removes a path from a document. If it's the last path, the document is removed entirely.
  /// @param name - The name of the document.
  /// @param path - The path to remove.
  #[wasm_bindgen(js_name = removePath)]
  pub fn remove_path(&mut self, name: &str, path: &str) -> Result<(), JsValue> {
    if let Some(doc) = self.documents.get_mut(name) {
      doc.paths.remove(path);
      if doc.paths.is_empty() {
        self.documents.remove(name);
        // Note: We can't easily remove the document from the corpus,
        // so we'll leave it there. This might lead to some inaccuracies over time.
      }
      Ok(())
    } else {
      Err(JsValue::from_str(&format!("Document '{}' not found", name)))
    }
  }

  /// Sets the similarity threshold for matching documents.
  /// @param threshold - The new similarity threshold (0.0 to 1.0).
  #[wasm_bindgen(js_name = setSimilarityThreshold)]
  pub fn set_similarity_threshold(&mut self, threshold: f32) {
    self.similarity_threshold = threshold;
  }

  /// Gets the current similarity threshold.
  /// @returns The current similarity threshold.
  #[wasm_bindgen(js_name = getSimilarityThreshold)]
  pub fn get_similarity_threshold(&self) -> f32 {
    self.similarity_threshold
  }

  /// Searches for documents similar to the given query.
  /// @param query - The search query.
  /// @returns An array of SearchResult objects representing the search results.
  #[wasm_bindgen(js_name = search)]
  pub fn search(&self, query: &str) -> Result<SearchResultArrayJS, JsValue> {
    console::log_1(&format!("Searching for: {}", query).into());
    console::log_1(&format!("Number of documents: {}", self.documents.len()).into());

    let query_vector = self.calculate_vector(query);

    log_vector_stats("Query", &query_vector);

    let mut results = Vec::new();
    for doc in self.documents.values() {
      console::log_1(&format!("Comparing with document: {}", doc.name).into());
      log_vector_stats(&format!("Document '{}'", doc.name), &doc.ngram_vector);

      let bm25_similarity = self.bm25_score(query, doc);
      let cosine_sim = cosine_similarity(&query_vector, &doc.ngram_vector);
      let combined_similarity = bm25_similarity * 0.7 + cosine_sim * 0.3;

      console::log_1(&format!("BM25 similarity: {}, Cosine similarity: {}, Combined similarity: {}",
                              bm25_similarity, cosine_sim, combined_similarity).into());

      if combined_similarity > 0.1 {
        results.push(SearchResult {
          similarity: combined_similarity,
          name: doc.name.clone(),
          paths: doc.paths.clone(),
        });
      }
    }

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    console::log_1(&format!("Number of results: {}", results.len()).into());

    serde_wasm_bindgen::to_value(&results)
      .map(SearchResultArrayJS::from)
      .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
  }


  /// Gets the current index data as a JSON-serializable object.
  /// @returns A JSON-serializable representation of the current index data.
  #[wasm_bindgen(js_name = getIndexData)]
  pub fn get_index_data(&self) -> Result<IndexDataJS, JsValue> {
    let serializable_documents: HashMap<String, SerializableDocument> = self.documents
      .iter()
      .map(|(name, doc)| (name.clone(), SerializableDocument {
        name: doc.name.clone(),
        paths: doc.paths.iter().cloned().collect(),
        word_frequencies: doc.word_frequencies.clone(),
        compressed_ngram_vector: compress_ngram_vector(&doc.ngram_vector),
      }))
      .collect();

    serde_wasm_bindgen::to_value(&serializable_documents)
      .map(IndexDataJS::from)
      .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
  }

  /// Creates a VectorizationSystem from serialized index data.
  /// @param index_data - The serialized index data.
  /// @returns A new VectorizationSystem instance.
  #[wasm_bindgen(js_name = createFromIndexData)]
  pub fn create_from_index_data(index_data: IndexDataJS) -> Result<VectorizationSystem, JsValue> {
    let serializable_documents: HashMap<String, SerializableDocument> = serde_wasm_bindgen::from_value(index_data.into())
      .map_err(|e| JsValue::from_str(&format!("Deserialization error: {}", e)))?;

    let mut system = VectorizationSystem::new();
    let mut corpus = CorpusBuilder::new().arity(2).pad_full(Pad::Auto).finish();

    for (name, doc) in serializable_documents {
      let indexed_doc = IndexedDocument {
        name: doc.name.clone(),
        paths: doc.paths.into_iter().collect(),
        word_frequencies: doc.word_frequencies.clone(),
        ngram_vector: decompress_ngram_vector(&doc.compressed_ngram_vector),
      };

      system.documents.insert(name.clone(), indexed_doc);

      // Add words to corpus
      for word in doc.word_frequencies.keys() {
        corpus.add_text(word);
      }
    }

    system.corpus = corpus;
    console::log_1(&format!("Deserialized {} documents", system.documents.len()).into());

    Ok(system)
  }

  fn generate_ngrams(&self, text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut ngrams = Vec::new();

    for word in &words {
      ngrams.push(word.to_string());
    }

    for window in words.windows(2) {
      ngrams.push(window.join(" "));
    }

    for window in words.windows(3) {
      ngrams.push(window.join(" "));
    }

    if words.len() <= 4 {
      ngrams.push(words.join(" "));
    }

    ngrams
  }

  fn calculate_vector(&self, text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_SIZE];
    let total_docs = self.documents.len().max(1) as f32;
    let ngrams = self.generate_ngrams(text);

    let mut term_frequency = HashMap::new();
    for ngram in &ngrams {
      *term_frequency.entry(ngram).or_insert(0) += 1;
    }

    for (ngram, &tf) in term_frequency.iter() {
      let doc_freq = self.documents.values()
        .filter(|doc| doc.word_frequencies.contains_key(&ngram.to_lowercase()))
        .count().max(1) as f32;
      let idf = (total_docs / doc_freq).ln() + 1.0;
      let tf_idf = (tf as f32) * idf;

      let weight = if ngram.split_whitespace().count() == 1 { 2.0 } else { 1.0 };

      let index = self.hash(ngram);
      vector[index] += tf_idf * weight;
    }

    normalize_vector(&mut vector);
    vector
  }

  fn bm25_score(&self, query: &str, doc: &IndexedDocument) -> f32 {
    let query_terms: Vec<&str> = query.split_whitespace().collect();
    let doc_length: f32 = doc.word_frequencies.values().sum::<u32>() as f32;
    let avg_doc_length: f32 = if self.documents.is_empty() {
      1.0
    } else {
      self.documents.values()
        .map(|d| d.word_frequencies.values().sum::<u32>() as f32)
        .sum::<f32>() / self.documents.len() as f32
    };

    query_terms.iter().map(|term| {
      let tf = *doc.word_frequencies.get(&term.to_lowercase()).unwrap_or(&0) as f32;
      let df = self.documents.values()
        .filter(|d| d.word_frequencies.contains_key(&term.to_lowercase()))
        .count().max(1) as f32;
      let idf = ((self.documents.len() as f32 - df + 0.5) / (df + 0.5)).ln().max(0.0);

      idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * doc_length / avg_doc_length.max(1.0)))
    }).sum()
  }

  fn hash(&self, token: &str) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut s = DefaultHasher::new();
    token.hash(&mut s);
    s.finish() as usize % EMBEDDING_SIZE
  }
}

#[derive(Serialize)]
struct SearchResult {
  similarity: f32,
  name: String,
  paths: HashSet<String>,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
  let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
  let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
  let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

  console::log_1(&format!("Dot product: {}, Norm A: {}, Norm B: {}", dot_product, norm_a, norm_b).into());

  if norm_a == 0.0 || norm_b == 0.0 {
    console::log_1(&"Warning: Zero norm detected".into());
    0.0
  } else {
    let similarity = dot_product / (norm_a * norm_b);
    console::log_1(&format!("Calculated similarity: {}", similarity).into());
    similarity
  }
}

fn normalize_vector(vector: &mut Vec<f32>) {
  let magnitude: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
  if magnitude > 0.0 {
    for i in 0..vector.len() {
      vector[i] /= magnitude;
    }
  } else {
    console::log_1(&"Warning: Zero magnitude vector".into());
  }
}

fn log_vector_stats(name: &str, vector: &[f32]) {
  let non_zero_count = vector.iter().filter(|&&x| x != 0.0).count();
  let sum: f32 = vector.iter().sum();
  let min: f32 = vector.iter().fold(f32::INFINITY, |a, &b| a.min(b));
  let max: f32 = vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
  console::log_1(&format!("{} vector stats - Non-zero elements: {}, Sum: {}, Min: {}, Max: {}",
                          name, non_zero_count, sum, min, max).into());
}

/// Initializes the panic hook for better error reporting in the browser console.
#[wasm_bindgen]
pub fn init() -> Result<(), JsValue> {
  console_error_panic_hook::set_once();
  Ok(())
}
