import streamlit as st
from streamlit_scroll_to_top import scroll_to_here
import json
import pyperclip
import random
from utils import *

st.set_page_config(layout="wide", page_title="LaTeX", page_icon=":scroll:")

if st.session_state.get("scroll_to_top", True):
    scroll_to_here(0, key='top')
    st.session_state.scroll_to_top = False
st.divider()
st.title("LaTeX Expressions Engine", anchor=False)
    
for k, v in st.session_state.items():
    if k.startswith("PST."):
        st.session_state[k] = v

if "saved_expressions" not in st.session_state:
    with open("expressions.json", 'r', encoding='utf-8') as file:
        st.session_state["saved_expressions"] = json.load(file)
    with open("examples.json", 'r', encoding='utf-8') as file:
        st.session_state["example_expressions"] = json.load(file)

if not st.session_state.get("refreshed", 0):
    st.session_state["refreshed"] = 1
    example_name, example_exp = random.choice(list(st.session_state["example_expressions"].items()))
    st.session_state["PST.EXP_NAME_INPUT"] = example_name
    st.session_state["PST.EXP_INPUT"] = example_exp

if st.session_state.get("EXP_INPUT_LOAD", 0):
    st.session_state["PST.EXP_NAME_INPUT"], st.session_state["PST.EXP_INPUT"] = st.session_state["EXP_INPUT_LOAD"]
    st.session_state["EXP_INPUT_LOAD"] = 0
    st.rerun()

with st.sidebar:
    st.title("Settings")
    st.divider()
    
    hide_preview = st.toggle("Hide Expression Preview", value=False)
    transparent_download = st.toggle("Transparent PNG", value=False)
    clear_on_save = st.toggle("Clear on Save", value=False)
    
    font_size = st.number_input("Font-size", value=14, min_value=8, max_value=72)
    
    st.download_button("Export Expressions",
                       json.dumps(st.session_state["saved_expressions"], indent=4),
                       file_name="expressions.json", type='primary',
                       mime="text/json",
                       icon=":material/download:",
                       use_container_width=True)
    
    if st.button("Import Expressions",
                 icon=":material/upload:",
                 type="primary",
                 use_container_width=True):
        pass
    

col1,col2 = st.columns([7,1])
col2.markdown("""<p style="min-height:12px"></p>""", unsafe_allow_html=True)
if col2.button("Load Example", use_container_width=True):
    st.session_state["EXP_INPUT_LOAD"] = random.choice(list(st.session_state["example_expressions"].items()))
    st.rerun()
    
expr_name = col1.text_input("Expression name:", key="PST.EXP_NAME_INPUT")
expr = st.text_area("LaTeX expression:", key="PST.EXP_INPUT", height=200).strip()

if not hide_preview:
    with st.container(border=True):
        st.latex(expr)
        if not expr: st.markdown(f"""<p style="margin-top:1rem;margin-bottom:2rem;font-size:{font_size+8}px;text-align:center">Expression preview will be shown here.</p>""", unsafe_allow_html=True)

col1,col2,col3 = st.columns([1,1,1])
if col1.button("Save Expression", type="primary", disabled=not expr_name or not expr, use_container_width=True):
    st.session_state["saved_expressions"].update({expr_name:expr})
    save_expressions()
    if clear_on_save:
        st.session_state["EXP_INPUT_LOAD"] = "", ""
        st.rerun()

if col2.button("Send to New Tab", disabled=not expr, use_container_width=True):
    load_exp_page(expr_name, expr, font_size)
    
if col3.button("Clear Canvas", disabled=not expr_name and not expr, use_container_width=True):
    st.session_state["EXP_INPUT_LOAD"] = "", ""
    st.rerun()

if st.session_state["saved_expressions"]:
    st.divider()
    st.header("Saved Expressions", anchor=False)
    st.markdown("""""", unsafe_allow_html=True)

del_exp = None
for name, expression in st.session_state["saved_expressions"].items():
    col1,col2,col3,col4 = st.columns([12,1,1,1])
    with col1.expander(name):
        st.latex(expression)
        if st.button("Send to New Tab", key=f"button_print_exp_{name}", type='primary', use_container_width=True):
            load_exp_page(name, expression, font_size)
        
    if col2.button("Copy", key=f"button_copy_expr_{name}", use_container_width=True):
        pyperclip.copy(expression)
        st.toast(f"\"{name}\" was copied to the clipboard!")
        
    if col3.button("Load", key=f"button_load_expr_{name}", use_container_width=True):
        st.session_state["EXP_INPUT_LOAD"] = name, expression
        st.session_state.scroll_to_top = True
        st.rerun()
        
    if col4.button("Delete", key=f"button_delete_expr_{name}", type="primary", use_container_width=True):
        st.toast(f"\"{name}\" was deletad!")
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