import streamlit as st
import json
import pyperclip

def save_expressions():
    with open("expressions.json", 'w', encoding='utf-8') as file:
        json.dump(st.session_state["saved_expressions"], file, indent=4)

st.set_page_config(layout="wide", page_title="LaTeX", page_icon=":scroll:")

st.title("Renderizador de LaTeX", anchor=False)

if "saved_expressions" not in st.session_state:
    with open("expressions.json", 'r', encoding='utf-8') as file:
        st.session_state["saved_expressions"] = json.load(file)

font_size = st.slider("Tamanho da fonte", min_value=10, max_value=40)

exp_name = st.text_input("Nome da Expressão")
exp = st.text_area("LaTeX aqui").strip()

if exp and st.button("Salvar Expressão", type="primary", disabled=not exp_name, use_container_width=True):
    st.session_state["saved_expressions"].update({exp_name:exp})
    save_expressions()

if exp and st.toggle("Preview", value=True): st.latex(exp)

if st.session_state["saved_expressions"]:
    st.divider()

    st.header("Expressões Salvas", anchor=False)
    st.markdown("""""", unsafe_allow_html=True)

del_exp = None
for name, expression in st.session_state["saved_expressions"].items():
    col1,col2,col3 = st.columns([8,1,1])
    with col1.expander(name):
        st.latex(expression)
        
    if col2.button("Copiar LaTeX", key=f"button_copy_exp_name_{name}", use_container_width=True):
        pyperclip.copy(expression)
        st.toast(f"Expressão \"{name}\" copiada!")
        
    if col3.button("Deletar", key=f"button_delete_exp_name_{name}", type="primary", use_container_width=True):
        st.toast(f"Expressão \"{name}\" deletada!")
        del_exp = name

if del_exp:
    st.session_state["saved_expressions"].pop(name)
    save_expressions()
    st.rerun()

st.markdown(f"""<style>
            .katex {{
                font-size: {font_size/10}em
            }}
            </style>""", unsafe_allow_html=True)
