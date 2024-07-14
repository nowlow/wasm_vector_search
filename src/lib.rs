use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

const EMBEDDING_SIZE: usize = 1000;
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.1;
const NGRAM_SIZES: [usize; 3] = [2, 3, 4];

/// Represents a document in the index with its associated metadata and vector representation.
#[derive(Serialize, Deserialize, Clone)]
struct IndexedDocument {
  name: String,
  paths: HashSet<String>,
  content: String,
  ngram_vector: Vec<f32>,
}

/// Contains all the indexed data, including documents and n-gram information.
#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
struct IndexData {
  documents: HashMap<String, IndexedDocument>,
  ngram_document_frequency: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize)]
struct SerializableDocument {
  name: String,
  paths: Vec<String>,
  ngram_vector: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct SerializableIndexData {
  documents: HashMap<String, SerializableDocument>,
  ngram_document_frequency: HashMap<String, usize>,
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export type IndexData = {
    documents: {
        [key: string]: {
            name: string;
            paths: string[];
            content: string;
            ngram_vector: Float32Array;
        }
    };
    ngram_document_frequency: { [key: string]: number };
};

export type MatchResult = {
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
        documents: HashMap::new(),
        ngram_document_frequency: HashMap::new(),
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
    let query_ngrams = self.generate_ngrams(query);
    let query_vector = self.calculate_vector(&query_ngrams);
    let mut results = Vec::new();

    for (name, doc) in &self.index_data.documents {
      let similarity = cosine_similarity(&query_vector, &doc.ngram_vector);
      if similarity > self.similarity_threshold {
        results.push(MatchResult {
          similarity,
          name: name.clone(),
          paths: doc.paths.clone(),
        });
      }
    }

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    serde_wasm_bindgen::to_value(&results)
        .map(MatchResultArrayJS::from)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
  }

  /// Adds or updates a document in the index.
  /// @param name - The name of the document.
  /// @param path - The path or identifier of the document.
  /// @param content - The content of the document.
  #[wasm_bindgen(js_name = addOrUpdateDocument)]
  pub fn add_or_update_document(&mut self, name: &str, path: &str, content: &str) {
    let new_ngrams = self.generate_ngrams(content);
    let old_ngrams = self.index_data.documents.get(name).map(|doc| self.generate_ngrams(&doc.content));
    self.update_ngram_data(&new_ngrams, old_ngrams.as_deref());
    let ngram_vector = self.calculate_vector(&new_ngrams);

    if let Some(doc) = self.index_data.documents.get_mut(name) {
      // Update existing document
      doc.paths.insert(path.to_string());
      doc.content = content.to_string();
      doc.ngram_vector = ngram_vector;
    } else {
      // Add new document
      let mut paths = HashSet::new();
      paths.insert(path.to_string());
      self.index_data.documents.insert(name.to_string(), IndexedDocument {
        name: name.to_string(),
        paths,
        content: content.to_string(),
        ngram_vector,
      });
    }
  }

  /// Adds a new path to an existing document.
  /// @param name - The name of the document.
  /// @param path - The new path to add.
  #[wasm_bindgen(js_name = addPath)]
  pub fn add_path(&mut self, name: &str, path: &str) -> Result<(), JsValue> {
    self.index_data.documents.get_mut(name)
        .map(|doc| {
          doc.paths.insert(path.to_string());
          Ok(())
        })
        .unwrap_or_else(|| Err(JsValue::from_str(&format!("Document '{}' not found", name))))
  }

  /// Updates the content of an existing document.
  /// @param name - The name of the document.
  /// @param content - The new content of the document.
  #[wasm_bindgen(js_name = updateContent)]
  pub fn update_content(&mut self, name: &str, content: &str) -> Result<(), JsValue> {
    let new_ngrams = self.generate_ngrams(content);
    let old_ngrams = self.index_data.documents.get(name).map(|doc| self.generate_ngrams(&doc.content));
    self.update_ngram_data(&new_ngrams, old_ngrams.as_deref());
    let ngram_vector = self.calculate_vector(&new_ngrams);

    self.index_data.documents.get_mut(name)
        .map(|doc| {
          doc.content = content.to_string();
          doc.ngram_vector = ngram_vector;
          Ok(())
        })
        .unwrap_or_else(|| Err(JsValue::from_str(&format!("Document '{}' not found", name))))
  }

  /// Removes a path from a document. If it's the last path, the document is removed entirely.
  /// @param name - The name of the document.
  /// @param path - The path to remove.
  #[wasm_bindgen(js_name = removePath)]
  pub fn remove_path(&mut self, name: &str, path: &str) -> Result<(), JsValue> {
    let document_removed = if let Some(doc) = self.index_data.documents.get_mut(name) {
      doc.paths.remove(path);
      doc.paths.is_empty()
    } else {
      return Err(JsValue::from_str(&format!("Document '{}' not found", name)));
    };

    if document_removed {
      if let Some(doc) = self.index_data.documents.remove(name) {
        let ngrams = self.generate_ngrams(&doc.content);
        self.update_ngram_data_for_removal(&ngrams);
      }
    }

    Ok(())
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

  /// Updates the n-gram data for the index.
  fn update_ngram_data(&mut self, new_ngrams: &[String], old_ngrams: Option<&[String]>) {
    if let Some(old_ngrams) = old_ngrams {
      for ngram in old_ngrams {
        if let Some(count) = self.index_data.ngram_document_frequency.get_mut(ngram) {
          *count -= 1;
          if *count == 0 {
            self.index_data.ngram_document_frequency.remove(ngram);
          }
        }
      }
    }

    for ngram in new_ngrams {
      *self.index_data.ngram_document_frequency.entry(ngram.clone()).or_insert(0) += 1;
    }
  }

  /// Updates the n-gram data when a document is removed.
  fn update_ngram_data_for_removal(&mut self, ngrams: &[String]) {
    for ngram in ngrams {
      if let Some(count) = self.index_data.ngram_document_frequency.get_mut(ngram) {
        *count -= 1;
        if *count == 0 {
          self.index_data.ngram_document_frequency.remove(ngram);
        }
      }
    }
  }

  /// Calculates the vector representation of a set of n-grams.
  fn calculate_vector(&self, ngrams: &[String]) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_SIZE];
    let total_docs = self.index_data.documents.len() as f32;

    for ngram in ngrams {
      if let Some(&doc_freq) = self.index_data.ngram_document_frequency.get(ngram) {
        let idf = (total_docs / (doc_freq as f32)).ln() + 1.0;
        let index = self.hash(ngram) % EMBEDDING_SIZE;
        vector[index] += idf;
      }
    }

    normalize_vector(&mut vector);
    vector
  }

  /// Generates n-grams from the input text.
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

  /// Hashes a token to an index in the embedding vector.
  fn hash(&self, token: &str) -> usize {
    let mut hash = 5381usize;
    for byte in token.bytes() {
      hash = ((hash << 5).wrapping_add(hash)).wrapping_add(byte as usize);
    }
    hash
  }

  /// Gets the current index data as a JSON-serializable object.
  /// @returns A JSON-serializable representation of the current index data.
  #[wasm_bindgen(js_name = getIndexData)]
  pub fn get_index_data(&self) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    struct SerializableIndexData {
      documents: Vec<SerializableDocument>,
      ngram_document_frequency: Vec<(String, usize)>,
    }

    #[derive(Serialize)]
    struct SerializableDocument {
      name: String,
      paths: Vec<String>,
      ngram_vector: Vec<f64>, // Changed to Vec<f64> for better precision
    }

    let serializable_documents: Vec<SerializableDocument> = self
        .index_data
        .documents
        .values()
        .map(|v| SerializableDocument {
          name: v.name.clone(),
          paths: v.paths.iter().cloned().collect(),
          ngram_vector: v.ngram_vector.iter().map(|&f| f as f64).collect(), // Convert f32 to f64
        })
        .collect();

    let serializable_ngram_frequency: Vec<(String, usize)> = self
        .index_data
        .ngram_document_frequency
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    let serializable_data = SerializableIndexData {
      documents: serializable_documents,
      ngram_document_frequency: serializable_ngram_frequency,
    };

    serde_wasm_bindgen::to_value(&serializable_data)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
  }

  #[wasm_bindgen(js_name = createFromIndexData)]
  pub fn create_from_index_data(index_data: JsValue) -> Result<VectorizationSystem, JsValue> {
    #[derive(Deserialize)]
    struct SerializableIndexData {
      documents: Vec<SerializableDocument>,
      ngram_document_frequency: Vec<(String, usize)>,
    }

    #[derive(Deserialize)]
    struct SerializableDocument {
      name: String,
      paths: Vec<String>,
      ngram_vector: Vec<f64>,
    }

    let serializable_data: SerializableIndexData = serde_wasm_bindgen::from_value(index_data)
        .map_err(|e| JsValue::from_str(&format!("Deserialization error: {}", e)))?;

    let documents = serializable_data.documents.into_iter().map(|v| {
      (v.name.clone(), IndexedDocument {
        name: v.name,
        paths: v.paths.into_iter().collect(),
        content: String::new(),
        ngram_vector: v.ngram_vector.into_iter()
            .map(|f| f as f32)
            .collect(),
      })
    }).collect();

    let ngram_document_frequency = serializable_data.ngram_document_frequency.into_iter().collect();

    Ok(VectorizationSystem {
      index_data: IndexData {
        documents,
        ngram_document_frequency,
      },
      similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
    })
  }
}

/// Initializes the panic hook for better error reporting in the browser console.
#[wasm_bindgen]
pub fn init() -> Result<(), JsValue> {
  console_error_panic_hook::set_once();
  Ok(())
}

/// Represents a match result from a search query.
#[derive(Serialize, Deserialize)]
struct MatchResult {
  similarity: f32,
  name: String,
  paths: HashSet<String>,
}

/// Calculates the cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
  let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
  let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
  let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
  dot_product / (norm_a * norm_b)
}

/// Normalizes a vector to unit length.
fn normalize_vector(vector: &mut Vec<f32>) {
  let magnitude: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
  if magnitude > 0.0 {
    for i in 0..vector.len() {
      vector[i] /= magnitude;
    }
  }
}
