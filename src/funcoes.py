"""Funções pequenas usadas pelos dois experimentos didáticos."""

from pathlib import Path
import pickle

import numpy as np


def aplicar_filtro_iir(x, b, a):
    """Aplica manualmente a equação de diferenças de um filtro IIR.

    Esta é a forma direta da equação. Ela é didática, mas filtros IIR de
    ordem alta podem ser menos estáveis numericamente nessa representação.
    """
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)

    if a[0] == 0:
        raise ValueError("a[0] não pode ser zero em um filtro IIR.")

    y = np.zeros(len(x), dtype=float)

    for n in range(len(x)):
        # Parte de avanço: entrada atual e entradas passadas.
        for k in range(len(b)):
            if n - k >= 0:
                y[n] += b[k] * x[n - k]

        # Realimentação: saídas passadas também influenciam a saída atual.
        for k in range(1, len(a)):
            if n - k >= 0:
                y[n] -= a[k] * y[n - k]

        y[n] = y[n] / a[0]

    if not np.all(np.isfinite(y)):
        raise FloatingPointError(
            "A forma direta do IIR ficou numericamente instável. "
            "A implementação manual foi mantida para estudo."
        )

    return y


def calcular_resposta_frequencia(b, a, fs, quantidade_pontos=2000):
    """Calcula H(e^jw) diretamente, sem usar scipy.signal.freqz."""
    frequencias = np.linspace(0, fs / 2, quantidade_pontos)
    resposta = []

    for f in frequencias:
        # Converte Hz para frequência angular digital, em radianos/amostra.
        w = 2 * np.pi * f / fs
        numerador = 0j
        denominador = 0j

        for k in range(len(b)):
            numerador += b[k] * np.exp(-1j * w * k)

        for k in range(len(a)):
            denominador += a[k] * np.exp(-1j * w * k)

        resposta.append(numerador / denominador)

    resposta = np.asarray(resposta)
    ganho_db = 20 * np.log10(np.abs(resposta) + 1e-12)
    return frequencias, ganho_db


def ganho_em_frequencia(f_alvo, frequencias, ganho_db):
    """Retorna o ganho no ponto calculado mais próximo da frequência alvo."""
    indice = np.argmin(np.abs(frequencias - f_alvo))
    return ganho_db[indice]


def calcular_spr(x, x_adv):
    """Calcula Signal-to-Perturbation Ratio (SPR), em dB."""
    energia_sinal = np.sum(np.abs(x) ** 2)
    energia_perturbacao = np.sum(np.abs(x - x_adv) ** 2)

    if energia_perturbacao == 0:
        return np.inf

    return 10 * np.log10(energia_sinal / energia_perturbacao)


def calcular_ober(delta, fs, fc):
    """Calcula a porcentagem de energia da perturbação fora de +/- fc.

    OBER significa Out-of-Band Energy Ratio. Nesta reprodução ele é mostrado
    em porcentagem, para facilitar a interpretação visual.
    """
    espectro = np.fft.fft(delta)
    frequencias = np.fft.fftfreq(len(delta), d=1 / fs)
    potencia = np.abs(espectro) ** 2

    mascara_fora = np.abs(frequencias) > fc
    energia_total = np.sum(potencia)
    energia_fora = np.sum(potencia[mascara_fora])

    if energia_total == 0:
        return 0.0

    return 100 * energia_fora / energia_total


def _chave_texto(chave):
    """Converte chaves textuais de pickles antigos para texto Python 3."""
    if isinstance(chave, bytes):
        return chave.decode("utf-8")
    return chave


def carregar_amostra_radioml(caminho, modulacao="QPSK", snr=18, indice=0):
    """Carrega uma janela (I, Q) do RadioML2016.10a.

    O arquivo usual do dataset é um dicionário cuja chave é (modulação, SNR)
    e cujo valor possui formato aproximado (quantidade, 2, 128).
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{caminho}'. "
            "Coloque RML2016.10a_dict.pkl dentro da pasta data/."
        )

    # encoding='latin1' permite abrir o pickle original, criado em Python 2.
    with caminho.open("rb") as arquivo:
        dados = pickle.load(arquivo, encoding="latin1")

    chave_encontrada = None
    for chave in dados:
        if not isinstance(chave, tuple) or len(chave) != 2:
            continue
        modulacao_chave = _chave_texto(chave[0])
        if modulacao_chave == modulacao and chave[1] == snr:
            chave_encontrada = chave
            break

    if chave_encontrada is None:
        chaves_disponiveis = list(dados.keys())[:10]
        raise KeyError(
            f"Não foi encontrada a chave ({modulacao!r}, {snr}). "
            f"Exemplos de chaves disponíveis: {chaves_disponiveis}"
        )

    amostras = np.asarray(dados[chave_encontrada])
    if amostras.ndim != 3 or amostras.shape[1:] != (2, 128):
        raise ValueError(
            "Formato inesperado no dataset. Era esperado "
            f"(quantidade, 2, 128), mas foi recebido {amostras.shape}."
        )
    if not 0 <= indice < len(amostras):
        raise IndexError(
            f"Índice {indice} inválido. Há {len(amostras)} amostras disponíveis."
        )

    return amostras[indice]
