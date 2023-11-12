use std::path::Path;

pub use orca::llm::bert::Bert;
pub use orca::llm::quantized::Quantized;
pub use orca::qdrant::Qdrant;

use orca::{
    llm::Embedding,
    prompts,
    record::{pdf::Pdf, Record, Spin},
};

pub fn set_font_env_var() {
    std::env::set_var("STANDARD_FONTS", ".\\pdf_fonts\\");
}

pub fn split_pdfs(filename: &str, length: usize) -> Vec<Record> {
    Pdf::from_file(filename, false)
        .spin()
        .unwrap()
        .split(length)
}

pub async fn vectordb_with_collection(database_ip: &str, collection_name: &str) -> Qdrant {
    let qdrant = Qdrant::new(database_ip);

    if qdrant.create_collection(collection_name, 384).await.is_ok() {
        qdrant
    } else {
        qdrant.delete_collection(collection_name).await.unwrap();
        qdrant
            .create_collection(collection_name, 384)
            .await
            .unwrap();
        qdrant
    }
}

pub fn get_stem_name(filename: &str) -> &str {
    Path::new(filename).file_stem().unwrap().to_str().unwrap()
}

pub async fn embed_and_store(
    vector_db: &Qdrant,
    collection_name: &str,
    model: &Bert,
    chunks: Vec<Record>,
) {
    let embeddings = model.generate_embeddings(prompts!(&chunks)).await.unwrap();
    vector_db
        .insert_many(collection_name, embeddings.to_vec2().unwrap(), chunks)
        .await
        .unwrap();
}
