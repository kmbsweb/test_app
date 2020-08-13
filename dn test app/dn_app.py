
import streamlit as st
import numpy as np
import pandas as pd
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

st.set_option('deprecation.showfileUploaderEncoding', False)

st.subheader('■特定ファイルからのLasso回帰係数の抽出')
st.write('本アプリケーションでは、指定形式のファイル（列名と列数はy、x1、x2、x3、x4、x5、x6、x7、x8）から回帰式の係数をシートごとに抽出します。Lasso回帰を用いますが次式の関数Jを最小化する回帰係数wを求めています。')
st.latex(r'''
J(w) = MSE(w)+α\sum_{i=1}^{n}|w_i|
''')
st.markdown("""・最適な λ を指定した交差検証の回数により、10^-6 ≤ α ≤ 10^2 の範囲内で 0.1 間隔のグリッドサーチを実施。<br>
    ・欠損値がある場合は、1つ前の値で補完<br>
    ・Scikit LearnのLassoCVを使用
    """, unsafe_allow_html=True)

st.sidebar.markdown('Set Parameter')
nholds = st.sidebar.number_input('交差検証回数',min_value=1,max_value=30,value=10)
posneg = st.sidebar.checkbox('係数符号：正', value=False)


uploaded_file = st.file_uploader("Choose a Excel file", type="xlsx")

if uploaded_file is not None:
    file = pd.ExcelFile(uploaded_file)
    #if uploaded_file is not None:
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    sheet_names = file.sheet_names

    dx = pd.DataFrame()

    for name in sheet_names:
        df = file.parse(name)

        X = df.loc[:,['x1','x2','x3','x4','x5','x6','x7','x8']]
        Y = df['y']

        scaler = StandardScaler()
        clf = LassoCV(alphas=10 ** np.arange(-6, 1, 0.1), positive=posneg, cv=nholds)

        scaler.fit(X)
        clf.fit(scaler.transform(X.fillna(method = 'ffill')), Y)
    
        tx = pd.DataFrame(np.hstack([clf.alpha_, clf.intercept_, clf.coef_, ])).transpose()
        tx.columns = ['alpha','Intercept','x1','x2','x3','x4','x5','x6','x7','x8']

        dx = pd.concat([dx, tx])

    dx.index = sheet_names
    st.dataframe(dx) 

    option = st.selectbox('列名選択',list(dx.columns))
    chart_data = dx[[option]]

    st.line_chart(chart_data)

    csv = dx.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown('### **⬇️ Download output CSV File **')
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ".csv")'
    st.markdown(href, unsafe_allow_html=True)

