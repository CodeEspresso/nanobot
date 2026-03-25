FROM python:3.12-slim-bookworm

# 国内 apt 镜像
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# 国内 pip 镜像
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

WORKDIR /app

# 复制项目文件并安装
COPY pyproject.toml README.md LICENSE ./
COPY nanobot/ nanobot/
RUN mkdir -p bridge && \
    pip install --no-cache-dir . && \
    rm -rf bridge

# 创建配置目录
RUN mkdir -p /root/.nanobot

EXPOSE 18790

ENTRYPOINT ["nanobot"]
CMD ["gateway"]
