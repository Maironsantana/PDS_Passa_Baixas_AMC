# Reprodução didática do filtro FCIAA

Este projeto reproduz apenas a parte de filtragem passa-baixas do artigo
*Frequency-Constrained Iterative Adversarial Attacks for Automatic Modulation
Classification (FCIAA)*. Não há CNN, FGSM, PGD nem o algoritmo FCIAA completo.

O filtro é um Butterworth passa-baixas de ordem 15, com frequência de
amostragem de 200 kHz e frequência de corte de 15,6 kHz. A aplicação do filtro
IIR é implementada manualmente para deixar visível a equação de diferenças.

## Ambiente virtual e instalação

Para isolar as dependências do projeto, crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Com o ambiente ativado, use `python` para executar os experimentos. Para sair
do ambiente virtual ao terminar, execute `deactivate`.

## Experimento 1: sinal sintético

```bash
python experimento_1.py
```

O script gera um sinal com senoides de 5, 10, 30 e 60 kHz, filtra esse sinal e
salva em `results/`:

- `resposta_frequencia.png`
- `espectro_sintetico_antes_depois.png`

O ganho próximo de 15,6 kHz deve ficar aproximadamente em -3 dB.

## Experimento 2: sinal I/Q do RadioML

Coloque o arquivo fornecido `RML2016_10a_dict.pkl` neste caminho:

```text
data/RML2016_10a_dict.pkl
```

Em seguida, execute:

```bash
python experimento_2.py
```

O script usa uma amostra QPSK com SNR de 18 dB, cria uma perturbação complexa
sintética e filtra apenas essa perturbação. Os arquivos gerados são:

- `iq_tempo.png`
- `perturbacao_espectro_antes_depois.png`

Também são mostradas no terminal as métricas SPR e OBER. Os valores não devem
coincidir necessariamente com os do artigo, pois a perturbação usada aqui é
ruído sintético, e não o resultado do ataque iterativo FCIAA.

## Uso de inteligência artificial

Foi utilizado auxílio de IA como apoio à implementação inicial e à organização
do projeto. Com esse apoio, foram criados os scripts dos dois experimentos, as
funções compartilhadas para filtro IIR, FFT, SPR e OBER, os comentários
didáticos no código e esta documentação. A especificação do experimento e a
revisão do conteúdo permanecem sob responsabilidade do autor do projeto.
