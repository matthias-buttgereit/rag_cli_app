# Use a Rust base image
FROM rust:latest as builder

# Set the working directory in the container
WORKDIR /usr/src/myapp

# Copy the whole project to the container
COPY . .

# Build the application
RUN cargo build --release

# Use a slim image to run the application
FROM debian:12-slim

# Set the working directory
WORKDIR /usr/src/myapp

# Copy the built executable from the previous stage
COPY --from=builder /usr/src/myapp/target/release/rag_cli_app .

# Command to run your application
CMD ["./rag_cli_app"]
