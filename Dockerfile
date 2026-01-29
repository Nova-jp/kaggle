# Kaggle公式のPythonイメージを使用
FROM gcr.io/kaggle-images/python:latest

# 作業ディレクトリの設定
WORKDIR /work

# 必要なパッケージがあればここで追加インストール
# RUN pip install --upgrade pip

# Jupyter Labの設定（パスワードなし、全てのIPを許可）
COPY jupyter_notebook_config.py /root/.jupyter/

# コンテナ起動時にJupyter Labを立ち上げる
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
