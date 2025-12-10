# Movie Recommendation System

Movie recommendation service.

## Overview

This repository contains code for a movie recommendation system powered by AI.

## Setup

### Prerequisites

- Anaconda or Miniconda
- Docker and Docker Compose

### Environment Setup

1. **Create a conda environment:**

   ```bash
   conda create -n movie-rec-sys python=3.12
   ```

1. **Activate the environment:**

   ```bash
   conda activate movie-rec-sys
   ```

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Pre-commit Hook

To ensure code quality and consistent formatting, we use `pre-commit`.

1. **Install pre-commit:**

   ```bash
   pip install pre-commit
   ```

1. **Install the git hook scripts:**
   Run this command once after cloning the repository to set up the hooks:

   ```bash
   pre-commit install
   ```

   Now, `pre-commit` will run automatically on `git commit`. You can also run it manually against all files:

   ```bash
   pre-commit run --all-files
   ```

### Docker

- Make sure you have Docker and Docker Compose installed on your system.

- **Build the Docker image:**

  ```bash
  docker build -t movie-recommend-api .
  ```

- **Build and start the service using Docker Compose:**

  ```bash
  docker compose up -d
  ```

- **To stop the service:**

  ```bash
  docker compose down
  ```

## Project Structure

The project is organized as follows:

- **`src/config/`**: Contains configuration settings.
  - Environment variables defined in your `.env` file are loaded and managed here (e.g., API keys, project settings).
- **`src/routers/`**: Contains FastAPI router definitions.
  - This is where the API endpoints and request handling logic are defined.
- **`src/model/`**: Contains the inference model code.
  - Logic related to the LLM inference and recommendation engine resides here.
- **`data/`**: Contains the dataset files (CSV).
