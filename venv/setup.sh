mkdir -p ~/.streamlit/

echo "\
[general]
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
