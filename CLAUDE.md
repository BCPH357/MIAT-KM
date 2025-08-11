# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIAT-KM is a Neo4j-based RAG (Retrieval-Augmented Generation) knowledge management system that integrates Neo4j knowledge graphs, LangChain, and Ollama. The system extracts knowledge triplets from PDF and Markdown documents, builds knowledge graphs, and provides intelligent Q&A functionality powered by configurable LLMs.

## Core Architecture

The system follows a microservices architecture using Docker Compose with three main services:

- **Neo4j**: Knowledge graph database (ports 7475:7474 for UI, 7688:7687 for Bolt)
- **Ollama**: Local LLM service with GPU acceleration (port 11435:11434)  
- **App**: Python application containing the main logic

### Key Components

- `main.py`: Interactive menu system and application entry point
- `rag_system.py`: Core RAG logic integrating knowledge retrieval and answer generation
- `knowledge_retriever.py`: Knowledge retrieval with LangChain GraphCypherQAChain integration
- `sentence_triplet_extractor.py`: File-to-triplets extraction supporting PDF and Markdown files
- `import_to_neo4j.py`: Bulk triplet import to Neo4j
- `ollama_client.py`: Ollama API client for LLM communication
- `config.py`: Global configuration file for LLM models and system settings

## Development Commands

### Docker Operations
```bash
# Start all services
sudo docker-compose up -d

# Check service status
sudo docker-compose ps

# View logs
sudo docker-compose logs [service_name]

# Stop services
sudo docker-compose down

# Rebuild specific service
sudo docker-compose build app

# Restart specific service
sudo docker-compose restart app
```

### Container Access
```bash
# Access application container
sudo docker-compose exec app bash

# Access Neo4j container
sudo docker-compose exec neo4j bash

# Access Ollama container
sudo docker-compose exec ollama bash
```

### Application Execution
```bash
# Run main application
sudo docker-compose exec app python main.py

# Run specific components
sudo docker-compose exec app python sentence_triplet_extractor.py
sudo docker-compose exec app python import_to_neo4j.py
```

### Model Management
```bash
# Download LLM model (inside ollama container)
sudo docker-compose exec ollama ollama pull gemma3:12b

# List available models
sudo docker-compose exec ollama ollama list
```

## System Configuration

### Service Endpoints
- Neo4j Browser: http://localhost:7475 (neo4j/password123)
- Ollama API: http://localhost:11435
- Neo4j Bolt: bolt://localhost:7688

### Data Directories
- PDF input: `./app/data/pdf/`
- Markdown input: `./app/data/markdown/`
- Processed data: `./app/data/processed/`
- Neo4j data: Docker volume `neo4j_data`
- Ollama models: Docker volume `ollama_data`

### Global Configuration

The system uses `config.py` for centralized configuration management:

- **LLM Model**: Change `OLLAMA_MODEL` to switch between different models (default: "gemma3:12b")
- **Ollama URL**: Configure `OLLAMA_BASE_URL` for custom Ollama instances
- **Model Parameters**: Adjust temperature, token limits, and other LLM parameters
- **Data Paths**: Centralized path configuration for all data directories

### Dependencies
Core Python packages (see `app/requirements.txt`):
- neo4j: Neo4j database driver
- langchain: LLM application framework
- langchain-community: Community integrations
- PyPDF2/pypdf: PDF processing
- requests: HTTP client

## Development Workflow

1. **Code Changes**: Modify Python files in `./app/` (changes are live-mounted)
2. **Restart Service**: `sudo docker-compose restart app` to reload changes
3. **Add Dependencies**: Update `requirements.txt` and rebuild with `docker-compose build app`
4. **Test Changes**: Run `sudo docker-compose exec app python main.py`

## RAG System Usage

The system provides three main functions via interactive menu:

1. **Extract Triplets**: Process both PDF and Markdown files using configurable LLM
   - PDF files: Place in `/app/data/pdf/`
   - Markdown files: Place in `/app/data/markdown/`
   - Supports automatic Markdown syntax cleaning (code blocks, headers, links, etc.)
2. **Import to Neo4j**: Load extracted triplets into knowledge graph
3. **RAG Q&A**: Interactive questioning with LangChain integration

### Q&A Commands
- Direct question: Auto-retrieval and answer generation
- `langchain <question>`: Show detailed retrieval process
- `quit`/`exit`: Exit Q&A mode

## GPU Acceleration

The system supports NVIDIA GPU acceleration for Ollama. Ensure nvidia-container-toolkit is installed and configured for optimal performance with the Gemma3 12B model.

## Data Backup

```bash
# Backup Neo4j database
sudo docker-compose exec neo4j neo4j-admin database dump neo4j /data/neo4j.dump

# Backup processed data
cp -r app/data/processed/ backup/
```