# Reddit Opinion Mining

A sophisticated web application that performs sentiment analysis and opinion mining on Reddit discussions using natural language processing and machine learning techniques.

## Overview

This application analyzes Reddit discussions to extract:
- Sentiment analysis using BERT
- Key topic extraction using GPT-3.5
- Discussion summarization
- Comment hierarchies and relevance scoring

## Architecture

The system consists of:

- **Flask Backend** - Handles web requests and orchestrates the analysis pipeline
- **Natural Language Processing**
  - BERT for sentiment analysis
  - GPT-3.5 for keyword extraction and summarization
- **Data Storage** - PostgreSQL database for storing Reddit comments and analysis results
- **Containerization** - Docker for consistent deployment
- **CI/CD** - GitHub Actions for automated deployment to AWS Lightsail

## Key Features

- Real-time Reddit discussion analysis
- Sentiment classification (Positive/Neutral/Negative)
- Intelligent keyword extraction
- AI-powered discussion summarization
- Comment hierarchy visualization
- Rate limiting and CSRF protection
- SSL/TLS security

## Technologies

- Python 3.11
- Flask & Flask extensions
- PyTorch & Transformers
- OpenAI GPT-3.5
- PostgreSQL
- Docker
- GitHub Actions
- AWS Lightsail

## Project Status

This project is actively maintained and deployed in production. The live application is not open for public cloning or deployment as it contains proprietary implementations and sensitive configurations.

## **Live App**  

🔗 **Try it here:** [Opinion Mining](http://lumicore.duckdns.org:8080/)

## License

[MIT License](LICENSE)

Copyright (c) 2025 lumicore-aidi-2005