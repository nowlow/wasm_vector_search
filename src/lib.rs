use wasm_bindgen::prelude::*;
use ndarray::{Array1, ArrayView1};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

const EMBEDDING_SIZE: usize = 100;
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;

#[wasm_bindgen]
pub struct VectorizationSystem {
  vocabulary: HashMap<String, Array1<f32>>,
  document_frequency: HashMap<String, usize>,
  total_documents: usize,
  similarity_threshold: f32,
}

#[derive(Serialize, Deserialize)]
pub struct MatchResult {
  is_match: bool,
  similarity: f32,
}

#[wasm_bindgen]
impl VectorizationSystem {
  #[wasm_bindgen(constructor)]
  pub fn new() -> Self {
    web_sys::console::log_1(&"VectorizationSystem::new called".into());
    VectorizationSystem {
      vocabulary: HashMap::new(),
      document_frequency: HashMap::new(),
      total_documents: 0,
      similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
    }
  }

  pub fn add_document(&mut self, text: &str) {
    let tokens = self.tokenize(text);
    let unique_tokens: HashSet<_> = tokens.iter().cloned().collect();

    for token in unique_tokens {
      *self.document_frequency.entry(token).or_insert(0) += 1;
    }

    self.total_documents += 1;
  }

  pub fn text_to_vector(&mut self, text: &str) -> Vec<f32> {
    web_sys::console::log_1(&JsValue::from_str(&format!("text_to_vector called with text: {}", text)));
    let tokens = self.tokenize(text);
    let mut text_vector = Array1::<f32>::zeros(EMBEDDING_SIZE);
    let mut token_count = HashMap::new();

    for token in &tokens {
      *token_count.entry(token).or_insert(0) += 1;
    }

    for (token, count) in token_count {
      let tf = count as f32 / tokens.len() as f32;
      let idf = (self.total_documents as f32 / (self.document_frequency.get(token).unwrap_or(&1) + 1) as f32).ln() + 1.0;
      let tfidf = tf * idf;

      let token_vector = if let Some(vector) = self.vocabulary.get(token) {
        vector.clone()
      } else {
        let new_vector = self.generate_word_vector(token);
        self.vocabulary.insert(token.to_string(), new_vector.clone());
        new_vector
      };

      text_vector += &(&token_vector * tfidf);
    }

    let result = text_vector.to_vec();
    web_sys::console::log_1(&JsValue::from_str(&format!("text_to_vector returning vector of length: {}", result.len())));
    result
  }

  fn tokenize(&self, text: &str) -> Vec<String> {
    let words: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
    let mut tokens = words.clone();

    // Add bigrams
    for i in 0..words.len() - 1 {
      tokens.push(format!("{} {}", words[i], words[i + 1]));
    }

    tokens
  }

  fn generate_word_vector(&self, word: &str) -> Array1<f32> {
    let mut hasher = DefaultHasher::new();
    word.hash(&mut hasher);
    let seed = hasher.finish();

    Array1::from_vec((0..EMBEDDING_SIZE).map(|i| {
      let x = ((seed.wrapping_add(i as u64).wrapping_mul(1103515245).wrapping_add(12345)) % 2u64.pow(31)) as f32 / 2u64.pow(31) as f32;
      (x - 0.5) * 2.0 // This gives a value between -1 and 1
    }).collect())
  }

  pub fn is_match(&self, vector1: &[f32], vector2: &[f32]) -> JsValue {
    web_sys::console::log_1(&JsValue::from_str(&format!("is_match called with vector1 length: {}, vector2 length: {}", vector1.len(), vector2.len())));
    if vector1.len() != EMBEDDING_SIZE || vector2.len() != EMBEDDING_SIZE {
      web_sys::console::log_1(&JsValue::from_str(&format!("Vector length mismatch. Expected {}, got {} and {}", EMBEDDING_SIZE, vector1.len(), vector2.len())));
      return serde_wasm_bindgen::to_value(&MatchResult { is_match: false, similarity: 0.0 }).unwrap();
    }
    let v1 = ArrayView1::from(vector1);
    let v2 = ArrayView1::from(vector2);
    let similarity = cosine_similarity(&v1, &v2);
    web_sys::console::log_1(&JsValue::from_str(&format!("Similarity: {}", similarity)));
    let result = MatchResult {
      is_match: similarity > self.similarity_threshold,
      similarity,
    };
    serde_wasm_bindgen::to_value(&result).unwrap()
  }

  pub fn set_similarity_threshold(&mut self, threshold: f32) {
    web_sys::console::log_1(&JsValue::from_str(&format!("Setting similarity threshold to: {}", threshold)));
    self.similarity_threshold = threshold;
  }

  pub fn get_similarity_threshold(&self) -> f32 {
    self.similarity_threshold
  }
}

fn cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
  let dot_product = a.dot(b);
  let norm_a = a.dot(a).sqrt();
  let norm_b = b.dot(b).sqrt();
  dot_product / (norm_a * norm_b)
}

#[wasm_bindgen]
pub fn init() -> Result<(), JsValue> {
  console_error_panic_hook::set_once();
  Ok(())
}
