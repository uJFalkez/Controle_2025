import streamlit as st
import streamlit.components.v1 as components
import time

st.title("Renderizador LaTeX com botão bonitão")

expr = st.text_input("Digite a expressão LaTeX (sem cifrões):", r"\frac{d^2 f}{dx^2}")

if "abrir_latex" not in st.session_state:
    st.session_state.abrir_latex = False

if expr:
    st.latex(expr)

    if st.button("🔍 Abrir em tela cheia para print"):
        st.session_state.abrir_latex = True

    # Se flag ativada, injeta JS e reseta flag
    if st.session_state.abrir_latex:
        safe_expr = expr.replace("\\", "\\\\").replace('"', '\\"')
        nonce = int(time.time() * 1000)

        components.html(f"""
        <script>
            const win = window.open("", "_blank_{nonce}");
            win.document.write(`<!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{
                            background: white;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100vh;
                            font-size: 2.5em;
                        }}
                    </style>
                    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"><\\/script>
                    <script id="MathJax-script" async
                        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"><\\/script>
                </head>
                <body>
                    $$ {safe_expr} $$
                </body>
                </html>`);
            win.document.close();
        </script>
        """, height=0)

        st.session_state.abrir_latex = False
