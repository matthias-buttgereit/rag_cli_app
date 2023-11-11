use clap::Parser;
use orca::{
    llm::{bert::Bert, quantized::Quantized, Embedding},
    pipeline::simple::LLMPipeline,
    pipeline::Pipeline,
    prompt::context::Context,
};
use rag_cli_app::*;

const PROMPT_FOR_MODEL: &str = include_str!("../../prompt.txt");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(long)]
    file: String,

    #[clap(long)]
    prompt: String,
}

#[tokio::main]
async fn main() {
    set_font_env_var();

    let args = Args::parse();
    let collection_name = get_stem_name(&args.file);

    let pdf_records = split_pdfs(&args.file, 250);
    let qdrant = vectordb_with_collection("http://localhost:6334", collection_name).await;

    let bert = Bert::new().build_model_and_tokenizer().await.unwrap();
    embed_and_store(&qdrant, collection_name, &bert, pdf_records).await;

    let query_embedding = bert
        .generate_embedding(orca::prompt!(args.prompt))
        .await
        .unwrap();

    let result = qdrant
        .search(
            collection_name,
            query_embedding.to_vec().unwrap().clone(),
            7,
            None,
        )
        .await
        .unwrap();

    let context = serde_json::json!({
        "user_prompt": args.prompt,
        "payloads": result
            .iter()
            .filter_map(|found_point| {
                found_point.payload.as_ref().map(|payload| {
                    // Assuming you want to convert the whole payload to a JSON string
                    serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string())
                })
            })
            .collect::<Vec<String>>()
    });

    let mistral = Quantized::new()
        .with_model(orca::llm::quantized::Model::Mistral7bInstruct)
        .with_sample_len(150)
        .load_model_from_path("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        .unwrap()
        .build_model()
        .unwrap();

    let mut pipe = LLMPipeline::new(&mistral).with_template("query", PROMPT_FOR_MODEL);
    pipe.load_context(&Context::new(context).unwrap()).await;

    let response = pipe.execute("query").await.unwrap();
    println!("\nResponse: {}", response.content());
}
