FROM continuumio/miniconda3:latest

WORKDIR /app

COPY pyproject.toml README.md environment.yml ./
COPY mlops ./mlops

RUN conda env create -f environment.yml -y \
    && conda clean -a -y

CMD ["conda", "run", "--no-capture-output", "-n", "mlops", "python", "-m", "mlops.dataset"]
