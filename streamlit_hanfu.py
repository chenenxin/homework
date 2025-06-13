import streamlit as st
import sys
from PIL import Image
import pathlib
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from fastai.vision.all import *
import torch
import random

# 设置页面配置
st.set_page_config(
    page_title="汉服智能助手",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS 样式
st.markdown("""
<style>
    .stApp {
        background-image: url("data:image/svg+xml,%3Csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='ricePaper' width='200' height='200' patternUnits='userSpaceOnUse'%3E%3Cpath d='M 200 0 L 0 0 0 200' fill='none' stroke='%23f0e8d0' stroke-width='0.5' opacity='0.2'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='%23f9f6f0' fill-opacity='0.9' pattern='url(%23ricePaper)'/%3E%3C/svg%3E");
        padding: 30px 50px;
    }
    h1 {
        color: #6b3e00;
        border-bottom: 3px solid #d4c4a6;
        font-family: 'STSong', 'SimSun', serif;
    }
    .card {
        background: linear-gradient(145deg, #fffaf0, #f5f0e1);
        border-radius: 12px;
        padding: 30px;
        border: 1px solid #e8dcc3;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(145deg, #f8f3e9, #e9e0d2);
        padding-top: 20px;
    }
    [data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    [data-testid="stSidebar"] .stRadio label {
        background-color: #d4c4a6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 16px;
        box-shadow: 2px 2px 5px rgba(107, 62, 0, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #e1d3c0;
        box-shadow: 3px 3px 7px rgba(107, 62, 0, 0.4);
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] .stRadio input:checked + label {
        background-color: #a67c52;
        box-shadow: inset 2px 2px 5px rgba(107, 62, 0, 0.3);
        transform: translateY(1px);
    }
    .st-bb {
        background-color: #a67c52;
    }
    .equal-cols {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
    }
    .pred-result {
        font-size: 1.2rem;
        color: #6b3e00;
    }
    .interpretation {
        background-color: rgba(245, 235, 219, 0.8);
        border-left: 4px solid #d4a976;
        padding: 15px;
        border-radius: 6px;
    }
    .card-image {
        max-width: 50%;
        border-radius: 8px;
    }
    
    /* 自定义按钮样式 */
    button.stButton {
        /* 未点击状态 */
        background-color: #d4a976;
        background-image: linear-gradient(145deg, #d4a976, #a67c52);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 14px;
        box-shadow: 2px 2px 5px rgba(107, 62, 0, 0.3);
        transition: all 0.3s ease;
        font-weight: bold;
    }
    
    button.stButton:hover {
        /* 悬停状态 */
        background-image: linear-gradient(145deg, #e0b98e, #bf956b);
        box-shadow: 3px 3px 7px rgba(107, 62, 0, 0.4);
        transform: translateY(-1px);
    }
    
    button.stButton:active {
        /* 点击状态 */
        background-image: linear-gradient(145deg, #a67c52, #8c6845);
        box-shadow: inset 2px 2px 5px rgba(107, 62, 0, 0.3);
        transform: translateY(1px);
    }
    
    /* 特殊功能按钮样式（如侧边栏按钮） */
    [data-testid="stSidebar"] button.stButton {
        background-color: #d4c4a6;
        background-image: linear-gradient(145deg, #d4c4a6, #bcaa90);
    }
    
    [data-testid="stSidebar"] button.stButton:hover {
        background-image: linear-gradient(145deg, #e1d3c0, #cdc1a8);
    }
    
    [data-testid="stSidebar"] button.stButton:active {
        background-image: linear-gradient(145deg, #bcaa90, #a89880);
    }
    
    /* 已点击按钮的特殊样式 */
    button.stButton.clicked {
        background-image: linear-gradient(145deg, #8c6845, #6b4e33);
        box-shadow: inset 2px 2px 5px rgba(107, 62, 0, 0.5);
        transform: translateY(1px);
        color: #f8f3e9;
    }
    
    /* 表单提交按钮 */
    button[data-testid="stFormSubmitButton"] {
        background-color: #a67c52;
        background-image: linear-gradient(145deg, #a67c52, #8c6845);
        font-weight: bold;
    }
    
    button[data-testid="stFormSubmitButton"]:hover {
        background-image: linear-gradient(145deg, #bf956b, #a67c52);
    }
    
    button[data-testid="stFormSubmitButton"]:active {
        background-image: linear-gradient(145deg, #8c6845, #6b4e33);
    }
    
    /* 评分按钮 */
    div[data-testid="stRadio"] > div {
        display: flex;
        justify-content: center;
        gap: 5px;
    }
    
    div[data-testid="stRadio"] label {
        background-color: #f5f0e1;
        border: 1px solid #d4c4a6;
        border-radius: 4px;
        padding: 6px 12px;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    div[data-testid="stRadio"] label:hover {
        background-color: #e8dcc3;
    }
    
    div[data-testid="stRadio"] input:checked + label {
        background-color: #a67c52;
        color: white;
        border-color: #8c6845;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript 脚本用于管理按钮状态
st.markdown("""
<script>
    // 页面加载完成后执行
    document.addEventListener('DOMContentLoaded', function() {
        // 查找所有按钮
        const buttons = document.querySelectorAll('button.stButton');
        
        // 为每个按钮添加点击事件
        buttons.forEach(button => {
            button.addEventListener('click', function() {
                // 移除所有按钮的点击状态
                buttons.forEach(btn => btn.classList.remove('clicked'));
                // 为当前按钮添加点击状态
                this.classList.add('clicked');
                
                // 存储按钮状态到本地存储
                localStorage.setItem('clickedButton', this.id || this.textContent);
            });
            
            // 检查本地存储，恢复之前的点击状态
            const savedButton = localStorage.getItem('clickedButton');
            if (savedButton && (button.id === savedButton || button.textContent === savedButton)) {
                button.classList.add('clicked');
            }
        });
        
        // 表单提交按钮状态管理
        const formSubmitButtons = document.querySelectorAll('button[data-testid="stFormSubmitButton"]');
        formSubmitButtons.forEach(button => {
            button.addEventListener('click', function() {
                // 添加点击状态
                this.classList.add('clicked');
                
                // 2秒后自动移除点击状态（防止表单提交后按钮一直保持点击状态）
                setTimeout(() => {
                    this.classList.remove('clicked');
                }, 2000);
            });
        });
    });
</script>
""", unsafe_allow_html=True)

# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

# 模型加载函数
@st.cache_resource
def load_model():
    try:
        model_path = pathlib.Path(__file__).parent / "汉服_model.pkl"
        if sys.platform == "win32":
            model_path = str(model_path)
        model = load_learner(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# 数据加载函数
@st.cache_data
def load_experiment_data():
    try:
        ratings_df = pd.read_excel("问卷数据.xlsx") if pathlib.Path("问卷数据.xlsx").exists() else None
        hanfu_df = pd.read_excel("汉服整合.xlsx") if pathlib.Path("汉服整合.xlsx").exists() else None
        return ratings_df, hanfu_df
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None, None

# 调整图片大小
def resize_image(image, max_width=400):
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return image.resize((max_width, new_height), Image.LANCZOS)
    return image

# 汉服识别模块
def hanfu_recognition_module():
    st.markdown('<h1 style="text-align:center; font-size:3.5em; color: #6b3e00; font-weight:bold;">🔎 汉服识别系统</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="card" style="padding: 10px;">
            <h2 style="color: #6b3e00; margin-top: 0; font-size: 1.8em;">🌸 识别热门汉服</h2>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"], key="recognition_uploader")
        
        if uploaded_file is not None:
            try:
                original_image = Image.open(uploaded_file)
                resized_image = resize_image(original_image, max_width=350)
                
                st.markdown('<div class="card-image-container">', unsafe_allow_html=True)
                st.image(resized_image, caption="上传的图片", use_container_width=False)
                st.markdown('</div>', unsafe_allow_html=True)
                
                image = PILImage.create(np.array(resized_image))
                model = load_model()
                
                if model:
                    try:
                        pred, pred_idx, probs = model.predict(image)
                        st.markdown(f'<div class="pred-result">预测结果: {pred}; 概率: {probs[pred_idx]:.4f}</div>', unsafe_allow_html=True)
                        st.session_state.recognition_prediction = pred
                    except Exception as e:
                        st.error(f"识别失败: {str(e)}")
                else:
                    st.warning("模型加载失败，无法进行识别")
            except Exception as e:
                st.error(f"处理上传图片时出错: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="padding: 10px;">
            <h2 style="color: #6b3e00; margin-top: 0; font-size: 1.8em;">📚 文化解读</h2>
        """, unsafe_allow_html=True)
        
        prediction = st.session_state.get('recognition_prediction', None)
        
        if uploaded_file is not None:
            if prediction:
                interpretations = {
                    "直裾": """
                        <div style="font-size: 1.3em; padding: 12px;">
                        直裾，又被称为襜褕，这个说法来自《说文解字》， 衣襟裾为方直，区别于曲裾。裾就是指衣服的大襟。直裾下摆部份剪裁为垂直，衣裾在身侧或侧后方，没有缝在衣上的系带，由布质或皮革制的腰带固定。
                        </div>
                    """,
                    "马面裙": """
                        <div style="font-size: 1.4em; padding: 12px;">
                            <br><br>
                           &nbsp;&nbsp;&nbsp;&nbsp;
                            马面裙，又名“马面褶裙”， 中国古代汉族女子的主要裙式，前后里外共有四个裙门，两两重合，外裙门有装饰，内裙门装饰较少或无装饰，马面裙侧面打裥，裙腰多用白色布，取白头偕老之意，以绳或纽固结。
                        </div>
                    """,
                    "曲裾": """
                        <div style="font-size: 1.4em; padding: 12px;">
                            <br><br>
                           &nbsp;&nbsp;&nbsp;&nbsp;
                            曲裾，全称曲裾袍，考古报告称绕襟袍。属上下分裁式，归类于“深衣类汉服”，故又称曲裾深衣、绕襟深衣。其历史款式原型流行于先秦至汉代。
                        </div>
                    """,
                    "齐胸襦裙": """
                        <div style="font-size: 1.4em; padding: 12px;">
                           <br><br>
                           &nbsp;&nbsp;&nbsp;&nbsp;
                            齐胸襦裙是对隋唐五代时期特有的一种女子裙装的称呼。一般穿法为一件对襟衫衣搭配一条下裙，或者一件交领上襦搭配下裙，即称为齐胸襦裙。齐胸衫裙是中国汉服形制的的一种，汉晋以来裙子的裙腰束于腰上，而隋唐五代时期裙子的裙腰束得更高，很多都在胸上，一些服装史上多称之为高腰衫裙。根据现在人们对它的考证，一般改称之为齐胸衫裙，齐胸衫裙已有文物出土，新疆阿斯塔纳唐墓出土两条唐裙。由于一些商家误导，齐胸衫裙常常被叫错为齐胸襦裙，正确叫法是齐胸衫裙。
                        </div>
                    """,
                    "齐腰襦裙": """
                        <div style="font-size: 1.4em; padding: 12px;">
                           <br><br>
                           &nbsp;&nbsp;&nbsp;&nbsp;
                           齐腰襦裙为襦裙的一类，属于汉服。裙腰与腰部平齐，故名。齐腰襦裙的上襦可为交领或直领（对襟）。同高腰襦裙相比，齐腰襦裙更为常见。按上襦分，可分为交领齐腰襦裙、直领齐腰襦裙（对襟齐腰襦裙）。按穿着对象分，可分为女式齐腰襦裙、男式齐腰襦裙。
                        </div>
                    """,
                    "曳撒": """
                        <div style="font-size: 1.4em; padding: 12px;">
                           <br><br>
                           &nbsp;&nbsp;&nbsp;&nbsp;
                           曳撒（yì sǎn ），读法源自蒙古语，为“一色（shǎi）”变音；来自蒙语“质孙（jisum），元代服饰之一。本意是蒙古袍。后在明王朝所吸纳继承，作为骑射服装和宫廷侍卫服装被广泛运用到明朝皇室的日常生活中。明朝人王世贞在《觚不觚集》里讲过“胡服也，其短袖或无袖，而衣中断，其下有横褶，而下腹竖褶之。若袖长则为曳撒“。虽然是胡服，但由于在明代被大量的使用，而在现代的汉服运动中穿着的人众多，所以这里也把它作为一种汉服款式进行解说。
                        </div>
                    """,
                    "袄裙": """
                        <div style="font-size: 1.4em; padding: 12px;">
                           <br><br>
                           &nbsp;&nbsp;&nbsp;&nbsp;
                           袄裙，是对中国古代女子上身穿袄，下身穿裙的统称。裙袄着装，从唐代开始就有衣物疏记录，一直到民国。现代一般谈论袄裙时候指的是明代的裙袄着装。有人对于袄裙定义为上衣穿裙子外为袄裙，实际是比较错误模糊的定义，因为有明一代，衫子也照样外穿，东晋十六国陶俑也显示此时代襦是穿于裙外的。所以不能以是否外穿定义袄裙，而应该以上身穿袄，下身穿裙的基本语境来定义袄裙一词。“袄裙”一般指的是明代有里子的双层的上襦和下裙（裙子一般为单层）的服装。
                        </div>
                    """,
                    "圆领袍": """
                        <div style="font-size: 1.4em; padding: 12px;">
                           <br><br>
                           &nbsp;&nbsp;&nbsp;&nbsp;
                           圆领袍，唐宋时称为“上领”、明朝则称为“团领”、“盘领”或“圆领”，是中国古代传统服饰常见款式之一。圆领袍流行于隋唐，宋朝以后，圆领袍成为官员们的正式服装之一，在后来的明朝也被大量运用，明朝的圆领袍、配上补，成为了分辨官位阶级的最方便的方式。
                        </div>
                    """,
                    "褙子": """
                        <div style="font-size: 1.4em; padding: 12px;">
                           <br><br>
                           &nbsp;&nbsp;&nbsp;&nbsp;
                           褙子，又名背子、绰子、绣䘿，是中国传统服饰的一种，起于隋唐。褙子直领对襟，两侧从腋下起不缝合，多罩在其他衣服外穿着。流行于宋、明两朝。宋朝褙子直领对襟，两腋开叉，衣裾短者及腰，长者过膝。宋朝女性多以褙子内着抹胸为搭配。明朝褙子有宽袖褙子、窄袖褙子两种。
                        </div>
                    """

                }
                if prediction in interpretations:
                    st.markdown(interpretations[prediction], unsafe_allow_html=True)
                else:
                    st.info(f"暂无 {prediction} 的详细解读", icon="📖")
            else:
                st.info("请等待识别完成以获取文化解读", icon="📖")
        else:
            st.info("请上传汉服图片以获取文化解读", icon="📖")
        st.markdown('</div>', unsafe_allow_html=True)

# 初始化会话状态
def init_session_state():
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.current_step = 1
        st.session_state.selected_hanfu = []
        st.session_state.user_ratings = {}
        st.session_state.recommendations = []
        st.session_state.rec_ratings = {}
        st.session_state.rating_range = (1, 5)
        st.session_state.satisfaction = None
        st.session_state.current_module = None
        st.session_state.button_states = {}  # 用于存储按钮点击状态

# 显示随机汉服并收集评分
def display_random_hanfu():
    global hanfu_df
    
    if hanfu_df is None or not isinstance(hanfu_df, pd.DataFrame) or hanfu_df.empty:
        st.error("汉服数据未正确加载，请检查数据文件。")
        return
    
    required_columns = ['item_id', 'name']
    for col in required_columns:
        if col not in hanfu_df.columns:
            st.error(f"汉服数据缺少必要的列: {col}")
            return
    
    try:
        hanfu_df['item_id'] = pd.to_numeric(hanfu_df['item_id'], errors='coerce')
        hanfu_df = hanfu_df.dropna(subset=['item_id'])
    except Exception as e:
        st.error(f"处理汉服 ID 时出错: {e}")
        return
    
    valid_item_ids = hanfu_df['item_id'].dropna().unique().tolist()
    if not valid_item_ids:
        st.error("没有找到有效的汉服 ID")
        return
    
    if not st.session_state.selected_hanfu:
        try:
            st.session_state.selected_hanfu = random.sample(valid_item_ids, min(3, len(valid_item_ids)))
        except ValueError:
            st.error(f"可用汉服数量不足，只有 {len(valid_item_ids)} 个有效汉服")
            return
        st.session_state.user_ratings = {}

    st.markdown('<h1 style="text-align:left; color: #6b3e00;">请为以下汉服评分</h1>', unsafe_allow_html=True)
    
    form_key = f"hanfu_rating_form_{hash(tuple(st.session_state.selected_hanfu))}"
    with st.form(key=form_key):
        valid_selected = []
        for item_id in st.session_state.selected_hanfu:
            if item_id in valid_item_ids:
                valid_selected.append(item_id)
        
        if len(valid_selected) != len(st.session_state.selected_hanfu):
            st.warning(f"已移除无效的汉服 ID，当前有效汉服数量: {len(valid_selected)}")
            st.session_state.selected_hanfu = valid_selected
        
        cols = st.columns(len(valid_selected))
        for i, item_id in enumerate(valid_selected):
            try:
                mask = hanfu_df['item_id'] == item_id
                if mask.any():
                    name = hanfu_df.loc[mask, 'name'].iloc[0]
                    if pd.isna(name) or not str(name).strip():
                        name = f"汉服 (ID: {item_id})"
                else:
                    name = f"未知汉服 (ID: {item_id})"
            except Exception as e:
                name = f"数据异常: {e}"
            
            with cols[i]:
                st.write(f"**{name}**")
                rating_options = list(range(1, 6))
                rating_labels = [f"{i}分" for i in rating_options]
                
                default_idx = 4
                if item_id in st.session_state.user_ratings:
                    default_idx = rating_options.index(st.session_state.user_ratings[item_id])
                
                rating_index = st.radio(
                    f"为汉服评分",
                    options=range(len(rating_options)),
                    format_func=lambda x: rating_labels[x],
                    index=default_idx,
                    key=f"rating_{item_id}_{i}",
                    horizontal=True
                )
                
                st.session_state.user_ratings[item_id] = rating_options[rating_index]

        submitted = st.form_submit_button("提交评分", type="primary")
        if submitted:
            if len(st.session_state.user_ratings) < len(valid_selected):
                st.warning("请为所有汉服评分")
            else:
                st.success("评分已提交！")
                st.write("您的评分如下:")
                for item_id, rating in st.session_state.user_ratings.items():
                    try:
                        name = hanfu_df.loc[hanfu_df['item_id'] == item_id, 'name'].iloc[0]
                    except:
                        name = f"汉服 (ID: {item_id})"
                    st.write(f"{name}: {rating}分")

# 显示推荐结果
def display_recommendations():
    global hanfu_df
    if hanfu_df is None or not isinstance(hanfu_df, pd.DataFrame):
        st.error("汉服数据异常，无法生成推荐")
        return

    st.header("🎯 个性化推荐")
    
    # 为按钮添加唯一ID以跟踪状态
    button_key = "get_recommendations_button"
    if st.button("获取个性化推荐", type="primary", key=button_key):
        # 记录按钮状态
        st.session_state.button_states[button_key] = True
        
        if len(st.session_state.user_ratings) < 3:
            st.warning("请先为 3 个汉服评分")
            return

        with st.spinner("正在生成推荐..."):
            if 'item_id' not in hanfu_df.columns:
                st.error("汉服数据缺少 item_id 列，无法生成推荐")
                return
            item_ids = hanfu_df['item_id'].dropna().tolist()
            unrated_items = [item for item in item_ids if item not in st.session_state.user_ratings]
            
            if len(unrated_items) >= 5:
                recommendations = random.sample(unrated_items, 5)
            else:
                recommendations = random.sample(item_ids, min(5, len(item_ids)))

            formatted_recs = []
            for item_id in recommendations:
                try:
                    if pd.notna(item_id) and item_id in hanfu_df['item_id'].values:
                        name = hanfu_df[hanfu_df['item_id'] == item_id]['name'].values[0]
                        pred_rating = random.uniform(1, 5)
                        formatted_recs.append({
                            'item_id': item_id,
                            'name': name,
                            'predicted_rating': pred_rating
                        })
                except Exception as e:
                    st.warning(f"处理推荐项 {item_id} 时出错: {e}")

            st.session_state.recommendations = formatted_recs
            st.success("推荐生成成功！")

    if 'recommendations' in st.session_state and st.session_state.recommendations:
        st.subheader("为您推荐汉服")
        for idx, rec in enumerate(st.session_state.recommendations):
            try:
                with st.expander(f"推荐 {idx + 1} - 预测评分: {rec['predicted_rating']:.2f}"):
                    st.text_area(
                        f"Hanfu ID: {rec['item_id']}",
                        rec['name'],
                        height=150,
                        disabled=True,
                        key=f"rec_hanfu_{rec['item_id']}_{idx}"
                    )
                    
                    rating_options = list(range(1, 6))
                    rating_labels = [f"{i}分" for i in rating_options]
                    default_idx = 4
                    
                    if rec['item_id'] in st.session_state.rec_ratings:
                        default_idx = rating_options.index(int(st.session_state.rec_ratings[rec['item_id']]))
                    
                    rating_index = st.radio(
                        "您的实际评分",
                        options=range(len(rating_options)),
                        format_func=lambda x: rating_labels[x],
                        index=default_idx,
                        key=f"rec_rating_{rec['item_id']}_{idx}",
                        horizontal=True
                    )
                    
                    st.session_state.rec_ratings[rec['item_id']] = float(rating_options[rating_index])
            except Exception as e:
                st.error(f"显示推荐项时出错: {e}")

# 计算满意度
def calculate_satisfaction(ratings):
    if not ratings:
        return 0.0
    avg_rating = np.mean(list(ratings.values()))
    return ((avg_rating - 1) / 4) * 100

# 显示满意度结果
def display_satisfaction():
    # 为按钮添加唯一ID以跟踪状态
    button_key = "calculate_satisfaction_button"
    if st.button("计算推荐满意度", key=button_key):
        # 记录按钮状态
        st.session_state.button_states[button_key] = True
        
        if not st.session_state.rec_ratings:
            st.warning("请先对推荐汉服评分")
            return

        satisfaction = calculate_satisfaction(st.session_state.rec_ratings)
        st.header(f"推荐满意度：{satisfaction:.1f}%")

        if satisfaction >= 80:
            st.success("🎉 非常满意！")
        elif satisfaction >= 60:
            st.info("😊 推荐效果良好，我们会继续优化")
        elif satisfaction >= 30:
            st.warning("😕 一般，有待改进")
        else:
            st.warning("😞 很抱歉未达到您的预期")

# 汉服展示模块
def hanfu_display_module():
    st.markdown('<h1 style="text-align:center; font-size:2.5em; color: #6b3e00; font-weight:bold;">👗汉服款式展示</h1>', unsafe_allow_html=True)
    
    status = st.selectbox("请选择性别", ('女', '男'))
    
    if status == '女':
        # 示例图片路径（请替换为实际路径）
        try:
            Image1 = Image.open('display/曲裾.jpg')
            Image2 = Image.open('display/直裾.jpg')
            Image6 = Image.open('display/圆领袍.jpg')
            Image4 = Image.open('display/齐胸襦裙.jpg')
            Image5 = Image.open('display/齐腰襦裙.jpg')
            Image3 = Image.open('display/马面裙.jpg')
            Image7 = Image.open('display/袄裙.jpg')
            Image8 = Image.open('display/褙子.jpg')
            
            # 创建布局
            st.markdown("### 女性汉服款式")
            row1 = st.columns(4)
            with row1[0]:
                st.image(Image1, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">曲裾</div>', unsafe_allow_html=True)
            with row1[1]:
                st.image(Image2, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">直裾</div>', unsafe_allow_html=True)
            with row1[2]:
                st.image(Image6, width=200) 
                st.markdown('<div style="text-align:center; color: #6b3e00;">圆领袍</div>', unsafe_allow_html=True)
            with row1[3]:
                st.image(Image4, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">齐胸襦裙</div>', unsafe_allow_html=True)
            
            row2 = st.columns(4)
            with row2[0]:
                st.image(Image5, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">齐腰襦裙</div>', unsafe_allow_html=True)
            with row2[1]:
                st.image(Image3, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">马面裙</div>', unsafe_allow_html=True)  
            with row2[2]:
                st.image(Image7, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">袄裙</div>', unsafe_allow_html=True)
            with row2[3]:
                st.image(Image8, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">褙子</div>', unsafe_allow_html=True) 
            
            # 展示表格数据
            df = pd.DataFrame({
                'Name': ['曲裾', '直裾', '圆领袍', '齐胸襦裙', '齐腰襦裙', '马面裙', '袄裙', '褙子'],
                'description': [
                    '流行于秦汉时期的绕襟深衣，线条优美，端庄大方。',
                    '直襟的汉服款式，剪裁简洁，行动便利，适合日常穿着。',
                    '圆领窄袖的袍服，多为官员或士人穿着，庄重大气。',
                    '唐代流行的高腰裙装，将裙头系于胸上，尽显雍容华贵。',
                    '裙腰与腰部齐平的传统裙装，清新秀丽，穿着舒适。',
                    '明代特色裙装，前后有两个裙门，两侧褶裥，端庄稳重。',  
                    '上衣为袄，下裙搭配的传统服饰，保暖性好，适合秋冬季节。',
                    '直领对襟的长外衣，两侧开衩，潇洒飘逸，男女皆可穿着。'
                ]
            })
            st.table(df)
            
        except Exception as e:
            st.error(f"图片加载失败: {e}")
            st.write("请确保图片文件存在且路径正确")
    else:
        # 示例图片路径（请替换为实际路径）
        try:
            Image4 = Image.open('display/男曲裾.jpeg')
            Image5 = Image.open('display/曳撒.jpg')
            Image6 = Image.open('display/圆领袍.jpg')
            Image7 = Image.open('display/男直裾.jpg')
            Image9 = Image.open('display/男褙子.jpg')
            
            # 创建布局
            st.markdown("### 男性汉服款式")
            row1 = st.columns(5)
            with row1[0]:
                st.image(Image4, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">曲裾</div>', unsafe_allow_html=True)
            with row1[1]:
                st.image(Image5, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">曳撒</div>', unsafe_allow_html=True)
            with row1[2]:
                st.image(Image6, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">圆领袍</div>', unsafe_allow_html=True)
            with row1[3]:
                st.image(Image7, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">直裾</div>', unsafe_allow_html=True)
            with row1[4]:
                st.image(Image9, width=200)
                st.markdown('<div style="text-align:center; color: #6b3e00;">褙子</div>', unsafe_allow_html=True)
            
            # 展示表格数据
            df = pd.DataFrame({
                'Name': ['曲裾', '曳撒', '圆领袍', '直裾','褙子'],
                'description': [
                    '流行于秦汉时期的绕襟深衣，线条优美，端庄大方。',
                    '明代典型男装，交领右衽，两侧开衩，下摆有褶裥，兼具威严与飘逸。',
                    '圆领窄袖的袍服，多为官员或士人穿着，庄重大气。',
                    '直襟的汉服款式，剪裁简洁，行动便利，适合日常穿着。',
                    '直领对襟的长外衣，两侧开衩，潇洒飘逸，男女皆可穿着。'
                ]
            })
            st.table(df)
            
        except Exception as e:
            st.error(f"图片加载失败: {e}")
            st.write("请确保图片文件存在且路径正确")

# 汉服评分与推荐模块
def hanfu_rating_recommendation_module():
    display_random_hanfu()
    display_recommendations()
    display_satisfaction()

# 加载数据
ratings_df, hanfu_df = load_experiment_data()

# 初始化会话状态
init_session_state()

# 侧边导航栏
st.sidebar.title("🌖汉服智能小助手🌔")
selected_module = st.sidebar.radio(
    "选择模块",
    ["🏠首页", "🔎汉服识别", "👗汉服展示", "🌟汉服评分与推荐"]
)

# 显示首页信息
if selected_module == "🏠首页":
    st.markdown('<h1 style="text-align:center; font-size:3.5em; color: #6b3e00; font-weight:bold;">🙌🏻汉服识别和推荐系统</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="text-align:left;">
        <p style="font-size:1.2em;">欢迎使用汉服智能助手，这是一个集汉服识别、文化解读与个性化推荐于一体的系统。</p>
        <p style="font-size:1.2em;">通过侧边栏导航，您可以：</p>
        <ul style="text-align:left; margin-left:20px; font-size:1.1em;">
            <li>使用汉服识别系统上传图片并获取汉服类型及文化解读</li>
            <li>通过汉服推荐系统获取个性化汉服推荐</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
elif selected_module == "🔎汉服识别":
    hanfu_recognition_module()
elif selected_module == "👗汉服展示":
    hanfu_display_module()
elif selected_module == "🌟汉服评分与推荐":
    hanfu_rating_recommendation_module()