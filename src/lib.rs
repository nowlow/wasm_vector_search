use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

const EMBEDDING_SIZE: usize = 1000;
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.1;
const NGRAM_SIZES: [usize; 3] = [2, 3, 4];

#[derive(Serialize, Deserialize, Clone)]
struct IndexedDocument {
  path: String,
  ngram_vector: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct IndexData {
  document_vectors: Vec<IndexedDocument>,
  ngram_vocabulary: HashSet<String>,
  ngram_idf: HashMap<String, f32>,
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export type IndexData = {
    document_vectors: Array<{
        path: string;
        vector: Float32Array;
    }>;
    document_frequency: { [key: string]: number };
    total_documents: number;
    vocabulary: string[];
};

export type MatchResult = {
    is_match: boolean;
    similarity: number;
    path: string;
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

  #[wasm_bindgen(typescript_type = "MatchResult[]")]
  pub type MatchResultArrayJS;
}

/// VectorizationSystem provides methods for text vectorization and similarity search.
#[wasm_bindgen]
pub struct VectorizationSystem {
  index_data: IndexData,
  similarity_threshold: f32,
}

#[wasm_bindgen]
impl VectorizationSystem {
  /// Creates a new VectorizationSystem.
  /// @param index_data - Optional initial index data. If not provided, an empty index will be created.
  #[wasm_bindgen(constructor)]
  pub fn new(index_data: Option<IndexDataJS>) -> Result<VectorizationSystem, JsValue> {
    console_error_panic_hook::set_once();

    let index_data = match index_data {
      Some(js_index_data) => serde_wasm_bindgen::from_value(js_index_data.into())
          .map_err(|e| JsValue::from_str(&format!("Failed to parse index data: {}", e)))?,
      None => IndexData {
        document_vectors: Vec::new(),
        ngram_vocabulary: HashSet::new(),
        ngram_idf: HashMap::new(),
      },
    };

    Ok(VectorizationSystem {
      index_data,
      similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
    })
  }


  /// Searches for documents similar to the given query.
  /// @param query - The search query.
  /// @returns An array of MatchResult objects representing the search results.
  #[wasm_bindgen(js_name = search)]
  pub fn search(&self, query: &str) -> Result<MatchResultArrayJS, JsValue> {
    let query_vector = self.text_to_vector(query);
    let mut results = Vec::new();

    for doc in &self.index_data.document_vectors {
      let similarity = cosine_similarity(&query_vector, &doc.ngram_vector);
      if similarity > self.similarity_threshold {
        results.push(MatchResult {
          is_match: true,
          similarity,
          path: doc.path.clone(),
        });
      }
    }

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    serde_wasm_bindgen::to_value(&results)
        .map(MatchResultArrayJS::from)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
  }

  /// Converts text to a vector representation.
  /// @param text - The input text to vectorize.
  /// @returns A vector (Float32Array) representation of the input text.
  fn text_to_vector(&self, text: &str) -> Vec<f32> {
    let ngrams = self.generate_ngrams(text);
    let mut vector = vec![0.0; EMBEDDING_SIZE];

    for ngram in ngrams {
      if let Some(&idf) = self.index_data.ngram_idf.get(&ngram) {
        let index = self.hash(&ngram) % EMBEDDING_SIZE;
        vector[index] += idf;
      }
    }

    normalize_vector(&mut vector);
    vector
  }

  /// Adds a document to the index.
  /// @param path - The path or identifier of the document.
  /// @param text - The content of the document.
  #[wasm_bindgen(js_name = addDocument)]
  pub fn add_document(&mut self, path: &str, text: &str) {
    let ngrams = self.generate_ngrams(text);
    let unique_ngrams: HashSet<_> = ngrams.into_iter().collect();

    // Update vocabulary and document frequency
    for ngram in &unique_ngrams {
      self.index_data.ngram_vocabulary.insert(ngram.clone());
      *self.index_data.ngram_idf.entry(ngram.clone()).or_insert(0.0) += 1.0;
    }

    // Update IDF values
    let total_docs = (self.index_data.document_vectors.len() + 1) as f32;
    for (_, doc_frequency) in self.index_data.ngram_idf.iter_mut() {
      *doc_frequency = (total_docs / *doc_frequency).ln() + 1.0;
    }

    // Generate document vectors
    let ngram_vector = self.text_to_vector(text);
    self.index_data.document_vectors.push(IndexedDocument {
      path: path.to_string(),
      ngram_vector,
    });
  }


  /// Serializes the current index data.
  /// @returns The serialized IndexData object.
  #[wasm_bindgen(js_name = serialize)]
  pub fn serialize(&self) -> Result<IndexDataJS, JsValue> {
    serde_wasm_bindgen::to_value(&self.index_data)
        .map(IndexDataJS::from)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
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

  fn generate_ngrams(&self, text: &str) -> Vec<String> {
    let text = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();
    let mut ngrams = Vec::new();

    for &size in &NGRAM_SIZES {
      for window in chars.windows(size) {
        ngrams.push(window.iter().collect::<String>());
      }
    }

    ngrams
  }

  fn hash(&self, token: &str) -> usize {
    let mut hash = 5381usize;
    for byte in token.bytes() {
      hash = ((hash << 5).wrapping_add(hash)).wrapping_add(byte as usize);
    }
    hash
  }
}

#[wasm_bindgen]
pub fn init() -> Result<(), JsValue> {
  console_error_panic_hook::set_once();
  Ok(())
}

#[derive(Serialize, Deserialize)]
struct MatchResult {
  is_match: bool,
  similarity: f32,
  path: String,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
  let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
  let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
  let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
  dot_product / (norm_a * norm_b)
}

fn normalize_vector(vector: &mut Vec<f32>) {
  let magnitude: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
  if magnitude > 0.0 {
    for i in 0..vector.len() {
      vector[i] /= magnitude;
    }
  }
}
