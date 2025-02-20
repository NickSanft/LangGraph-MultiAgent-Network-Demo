ollama create -f .\llama3.2-modelfile.txt llama3.2
ollama create -f .\mistral-modelfile.txt mistral
ollama create -f .\llava-modelfile.txt codellama

ollama run llama3.2 /bye

ollama run mistral /bye

ollama run codellama /bye
