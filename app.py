import streamlit as st
import numpy as np
import joblib

# ---------- 加载模型 ----------
@st.cache_resource
def load_model():
    return joblib.load("RF.pkl")

model = load_model()

# ---------- 页面配置 ----------
st.set_page_config(page_title="淋巴结转移预测", layout="wide")
st.title("甲状腺结节淋巴结转移风险预测")
st.markdown("请在下方逐项填写特征，点击 **开始预测** 获取淋巴结转移概率。")

# ---------- 输入区块 ----------
st.header("1. 基本信息")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age (年龄)", 0, 120, 45)
    gender = st.radio("Gender (性别)", ["男", "女"], index=1)   # 0男 1女
    weight = st.number_input("Weight (体重 kg)", 0.0, 200.0, 65.0)
    height = st.number_input("Height (身高 cm)", 0.0, 250.0, 165.0)
    bmi = st.number_input("BMI", 0.0, 50.0, 23.0)
with c2:
    pass
with c3:
    pass

st.header("2. 病史与生活习惯")
c1, c2, c3 = st.columns(3)
with c1:
    malignant_tumor = st.radio("恶性瘤史", ["否", "是"], index=0)
    hyperthyroidism = st.radio("甲亢", ["否", "是"], index=0)
    hypothyroidism = st.radio("甲减", ["否", "是"], index=0)
    hyperlipidemia = st.radio("高脂血症", ["否", "是"], index=0)
    diabetes = st.radio("糖尿病", ["否", "是"], index=0)
with c2:
    hypertension = st.radio("高血压", ["否", "是"], index=0)
    smoking = st.radio("吸烟史", ["否", "是"], index=0)
    alcohol = st.radio("饮酒史", ["否", "是"], index=0)

st.header("3. 超声特征")
c1, c2, c3 = st.columns(3)
with c1:
    us_diameter = st.number_input("US diameter (直径 cm)", 0.0, 10.0, 2.0)
    us_capsule_invasion = st.radio("包膜侵犯", ["否", "是"], index=0)
    composition = st.radio("组成", ["囊性/囊实性", "实性"], index=1)   # 0囊性 1实性
    margin = st.radio("边缘", ["清晰", "不清"], index=0)               # 0清晰 1不清
    shape = st.radio("形状", ["规则", "不规则"], index=0)              # 0规则 1不规则
    taller_than_wide = st.radio("纵横比 >1", ["<1", ">1"], index=0)    # 0<1 1>1
with c2:
    hyperechoic = st.radio("有高回声", ["无", "有"], index=0)
    blood_flow = st.radio("血流信号", ["无", "有"], index=0)
    multifocality = st.radio("多灶性", ["否", "是"], index=0)
    bilaterality = st.radio("双侧性", ["否", "是"], index=0)
with c3:
    st.write("**回声类型（可多选，请至少选择一项）**")
    hypoechoic = st.checkbox("低回声", value=True)
    slightly_hypo = st.checkbox("稍低回声", value=False)
    isoechoic = st.checkbox("等回声", value=False)
    mix_echo = st.checkbox("混合回声", value=False)

st.header("4. 实验室检查")
c1, c2, c3 = st.columns(3)
with c1:
    uric_acid = st.number_input("Uric Acid (尿酸)", 0.0, 1000.0, 300.0)
    creatinine = st.number_input("Creatinine (肌酐)", 0.0, 600.0, 70.0)
    total_protein = st.number_input("Total Protein (总蛋白)", 0.0, 100.0, 70.0)
    albumin = st.number_input("Albumin (白蛋白)", 0.0, 50.0, 40.0)
    serum_calcium = st.number_input("Serum Calcium (血钙)", 0.0, 4.0, 2.3, step=0.01)
    serum_phosphorus = st.number_input("Serum Phosphorus (血磷)", 0.0, 4.0, 1.2, step=0.01)
with c2:
    rbc = st.number_input("RBC (红细胞)", 0.0, 30.0, 4.5)
    plt = st.number_input("PLT (血小板)", 0.0, 600.0, 250.0)
    neutrophil = st.number_input("Neutrophil (中性粒)", 0.0, 30.0, 4.0)
    lymphocyte = st.number_input("Lymphocyte (淋巴)", 0.0, 10.0, 2.0)
    monocyte = st.number_input("Monocyte (单核)", 0.0, 10.0, 0.5)
    eosinophil = st.number_input("Eosinophil (嗜酸)", 0.0, 10.0, 0.1)
    basophil = st.number_input("Basophil (嗜碱)", 0.0, 10.0, 0.05)
with c3:
    plr = st.number_input("PLR", 0.0, 1000.0, 120.0)
    mlr = st.number_input("MLR", 0.0, 1000.0, 10.0)
    elr = st.number_input("ELR", 0.0, 1000.0, 5.0)
    blr = st.number_input("BLR", 0.0, 1000.0, 1.0)
    nlr = st.number_input("NLR", 0.0, 1000.0, 2.0)
    sii = st.number_input("SII", 0.0, 1000.0, 500.0)
    globulin = st.number_input("Globulin (球蛋白)", 0.0, 50.0, 30.0)
    ag_ratio = st.number_input("A/G Ratio", 0.0, 4.0, 1.2, step=0.01)

st.header("5. 肿瘤位置")
c1, c2 = st.columns(2)
with c1:
    loc_isthmus = st.radio("位置：峡部", ["否", "是"], index=0)
with c2:
    loc_right = st.radio("位置：右侧", ["否", "是"], index=0)

# ---------- 组装特征向量（严格保持顺序） ----------
def get_features():
    yes_no = {"是": 1, "否": 0}
    gender_map = {"男": 0, "女": 1}
    comp_map = {"囊性/囊实性": 0, "实性": 1}
    margin_map = {"清晰": 0, "不清": 1}
    shape_map = {"规则": 0, "不规则": 1}
    taller_map = {"<1": 0, ">1": 1}
    have_map = {"无": 0, "有": 1}    # 新增这一行

    features = [
        age,                                    # 1. Age
        gender_map[gender],                     # 2. Gender
        yes_no[malignant_tumor],                # 3. malignant.tumor.history
        yes_no[hyperthyroidism],                # 4. Hyperthyroidism
        yes_no[hypothyroidism],                 # 5. Hypothyroidism
        yes_no[hyperlipidemia],                 # 6. hyperlipidemia
        yes_no[diabetes],                       # 7. Diabetes.Mellitus
        yes_no[hypertension],                   # 8. Hypertension
        yes_no[smoking],                        # 9. Smoking.history
        yes_no[alcohol],                        # 10. History.of.alcohol.consumption
        yes_no[multifocality],                  # 11. Multifocality
        yes_no[bilaterality],                   # 12. Bilaterality
        us_diameter,                            # 13. US.diameter
        yes_no[us_capsule_invasion],            # 14. US.capsule.invasion
        comp_map[composition],                  # 15. Composition
        margin_map[margin],                     # 16. Margin
        shape_map[shape],                       # 17. Shape
        have_map[hyperechoic],                  # 19. Hyperechoic   ← 改这里
        have_map[blood_flow],                   # 20. US.blood.flow.signal  ← 改这里
        yes_no[blood_flow],                     # 20. US.blood.flow.signal
        uric_acid,                              # 21. Uric.Acid
        creatinine,                             # 22. Creatinine
        total_protein,                          # 23. Total.Protein
        albumin,                                # 24. Albumin
        serum_calcium,                          # 25. Serum.Calcium
        serum_phosphorus,                       # 26. Serum.Phosphorus
        rbc,                                    # 27. RBC
        plt,                                    # 28. PLT
        neutrophil,                             # 29. Neutrophil
        lymphocyte,                             # 30. Lymphocyte
        monocyte,                               # 31. Monocyte
        eosinophil,                             # 32. Eosinophil
        basophil,                               # 33. Basophil
        weight,                                 # 34. Weight
        height,                                 # 35. Height
        plr,                                    # 36. PLR
        mlr,                                    # 37. MLR
        elr,                                    # 38. ELR
        blr,                                    # 39. BLR
        nlr,                                    # 40. NLR
        sii,                                    # 41. SII
        bmi,                                    # 42. BMI
        globulin,                               # 43. Globulin
        ag_ratio,                               # 44. A.G.Ratio
        int(hypoechoic),                        # 45. US.echo.hypoechoic
        int(slightly_hypo),                     # 46. US.echo.Slightly.hypoechoic
        int(isoechoic),                         # 47. US.echo.isoechoic
        int(mix_echo),                          # 48. US.echo.mix.echoc
        yes_no[loc_isthmus],                    # 49. Location_isthmus
        yes_no[loc_right]                       # 50. Location_right
    ]
    return np.array(features).reshape(1, -1)

# ---------- 预测按钮 ----------
if st.button("🔍 开始预测"):
    input_vector = get_features()

    # 可选：检查回声类型至少选一个
    if not any([hypoechoic, slightly_hypo, isoechoic, mix_echo]):
        st.warning("⚠️ 您未选择任何回声类型，建议至少选择一项以获得合理预测。")

    try:
        pred_class = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        prob_met = proba[1]   # 淋巴结转移概率
        prob_no = proba[0]    # 无转移概率

        st.subheader("📊 预测结果")
        if pred_class == 1:
            st.error(f"预测分类：**淋巴结转移 (高风险)**")
        else:
            st.success(f"预测分类：**无淋巴结转移 (低风险)**")

        col_res1, col_res2 = st.columns(2)
        col_res1.metric("淋巴结转移概率", f"{prob_met:.2%}")
        col_res2.metric("无转移概率", f"{prob_no:.2%}")

        st.caption("⚠️ 本预测结果仅供科研参考，不能替代临床诊断。")

    except Exception as e:
        st.error(f"预测失败：{e}")