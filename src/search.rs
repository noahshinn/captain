use crate::embeddings::embedding;
use std::collections::BinaryHeap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Error embedding query")]
    EmbeddingError(#[from] reqwest::Error),
}

#[derive(Debug, Clone)]
pub struct EmbeddedDocument<'a, T> {
    pub embedding: &'a Vec<f32>,
    pub document: T,
}

#[derive(Debug, Clone)]
pub struct DenseEmbeddingSearchResult<'a, T> {
    pub embedded_document: &'a EmbeddedDocument<'a, T>,
    pub distance: f32,
}

impl<'a, T> PartialOrd for DenseEmbeddingSearchResult<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<'a, T> Ord for DenseEmbeddingSearchResult<'a, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a, T> PartialEq for DenseEmbeddingSearchResult<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<'a, T> Eq for DenseEmbeddingSearchResult<'a, T> {}

pub async fn dense_embedding_search<'a, T>(
    query: &str,
    embedded_documents: &'a [EmbeddedDocument<'a, T>],
    max_results: usize,
) -> Result<Vec<DenseEmbeddingSearchResult<'a, T>>, SearchError> {
    let query_embedding_result = embedding(vec![query.to_string()]).await.unwrap();
    let query_embedding = query_embedding_result.first().unwrap();
    let mut heap: BinaryHeap<DenseEmbeddingSearchResult<'a, T>> =
        BinaryHeap::with_capacity(max_results);
    for embedded_document in embedded_documents {
        let distance = cosine_distance(&query_embedding, &embedded_document.embedding);
        heap.push(DenseEmbeddingSearchResult {
            embedded_document: embedded_document,
            distance,
        });
    }
    let mut results = heap.into_sorted_vec();
    results.truncate(max_results);
    Ok(results)
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(a, b)| a * b).sum();
    let norm_a: f32 = a.iter().map(|a| a * a).sum();
    let norm_b: f32 = b.iter().map(|b| b * b).sum();
    dot_product / (norm_a * norm_b).sqrt()
}
