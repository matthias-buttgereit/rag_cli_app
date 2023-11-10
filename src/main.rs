use std::time::Instant;

use clap::Parser;
use orca::{
    llm::{bert::Bert, quantized::Quantized, Embedding},
    pipeline::simple::LLMPipeline,
    pipeline::Pipeline,
    prompt,
    prompt::context::Context,
    prompts,
    qdrant::Qdrant,
    record::{pdf::Pdf, Record, Spin},
};
// use serde_json::json;

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

    println!("FILE: {}\nPROMPT: {}", args.file, args.prompt);

    let pdf_records: Vec<Record> = Pdf::from_file(&args.file, false).spin().unwrap().split(250);
    let bert = Bert::new().build_model_and_tokenizer().await.unwrap();

    let collection_name = std::path::Path::new(&args.file)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("default_collection")
        .to_string();

    print!("Connecting to Qdrant ... ");
    let qdrant = Qdrant::new("http://localhost:6334");
    println!("CONNECTED!");

    print!("Collection '{}' ... ", collection_name);
    if qdrant
        .create_collection(&collection_name, 384)
        .await
        .is_ok()
    {
        println!("was newly created!")
    } else {
        println!("had to be deleted and created anew!");
        qdrant.delete_collection(&collection_name).await.unwrap();
        qdrant
            .create_collection(&collection_name, 384)
            .await
            .unwrap();
    }

    let embeddings = bert
        .generate_embeddings(prompts!(&pdf_records))
        .await
        .unwrap();
    qdrant
        .insert_many(&collection_name, embeddings.to_vec2().unwrap(), pdf_records)
        .await
        .unwrap();

    print!("Embedding the prompt ... ");
    let query_embedding = bert.generate_embedding(prompt!(args.prompt)).await.unwrap();
    println!("embedded!");

    print!("Receiving context ... ");
    let result = qdrant
        .search(
            &collection_name,
            query_embedding.to_vec().unwrap().clone(),
            7,
            None,
        )
        .await
        .unwrap();
    println!("received!");

    let prompt_for_model = r#"
{{#chat}}
    {{#system}}
    You are a highly advanced assistant. You receive a prompt from a user and relevant excerpts extracted from a PDF. You then answer truthfully to the best of your ability. If you do not know the answer, your response is I don't know.
    {{/system}}
    {{#user}}
    {{user_prompt}}
    {{/user}}
    {{#system}}
    Based on the retrieved information from the PDF, here are the relevant excerpts:
    {{#each payloads}}
    {{this}}
    {{/each}}
    Please provide a comprehensive answer to the user's question, integrating insights from these excerpts and your general knowledge.
    {{/system}}
{{/chat}}
        "#;

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

    println!("{:#?}", context);

    print!("Building model ... ");
    let mistral = Quantized::new()
        .with_model(orca::llm::quantized::Model::Mistral7bInstruct)
        .with_sample_len(150)
        .load_model_from_path("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        .unwrap()
        .build_model()
        .unwrap();
    println!("done!");

    print!("Loading context and prompt into template ... ");
    let mut pipe = LLMPipeline::new(&mistral).with_template("query", prompt_for_model);
    pipe.load_context(&Context::new(context).unwrap()).await;
    println!("done!");

    let start_time = Instant::now();

    println!("Crafting response ...");
    let response = pipe.execute("query").await.unwrap();

    let duration = start_time.elapsed().as_secs();
    println!("\nResponse: {}", response.content());

    println!(
        "\nGenerating this response took {}:{} minutes.",
        duration / 60,
        duration % 60
    );
}

fn set_font_env_var() {
    std::env::set_var("STANDARD_FONTS", ".\\pdf_fonts\\");
}
