# Plano de implementação — Reprodução do filtro do artigo FCIAA

## Contexto

Este projeto reproduz, de forma **simples e didática**, a parte de filtragem do artigo:

**Frequency-Constrained Iterative Adversarial Attacks for Automatic Modulation Classification (FCIAA)**

O foco não é reproduzir todo o algoritmo adversarial nem as redes neurais do artigo. O objetivo é entender e demonstrar o comportamento do filtro digital usado pelos autores.

Parâmetros reportados no artigo:

- Tipo: Butterworth passa-baixas
- Ordem: 15
- Frequência de amostragem: 200 kHz
- Frequência de Nyquist: 100 kHz
- Frequência de corte: 15,6 kHz
- Dataset utilizado no artigo: RadioML2016.10a

---

# Objetivo geral do agente

Ajude a implementar o projeto em **Python simples**, adequado para alguém iniciante em Processamento Digital de Sinais.

Prioridades:

1. Código fácil de ler.
2. Explicações dentro do próprio código.
3. Evitar classes e arquitetura excessivamente complexa.
4. Evitar funções prontas de processamento de sinais quando for razoável implementar a operação manualmente.
5. Usar NumPy e Matplotlib como ferramentas principais.
6. SciPy pode ser usado para **projetar os coeficientes do Butterworth**, pois derivar um Butterworth de ordem 15 manualmente foge do objetivo didático.
7. Sempre explicar o significado físico/matemático antes ou ao lado do código.

---

# Estrutura simples do projeto

Sugestão:

```text
fciaa_filter/
│
├── experimento_1.py
├── experimento_2.py
├── funcoes.py
├── data/
│   └── RML2016.10a_dict.pkl
├── resultados/
└── README.md
```

Se for possível manter tudo em apenas dois scripts e um arquivo de funções, prefira isso.

---

# Dependências

Usar apenas:

```text
numpy
matplotlib
scipy
```

Evitar pandas, seaborn e bibliotecas adicionais sem necessidade.

---

# Experimento 1 — Reprodução do filtro Butterworth

## Objetivo

Reproduzir o filtro descrito no artigo e entender sua resposta em frequência.

Parâmetros:

```python
FS = 200_000       # Hz
FC = 15_600        # Hz
ORDEM = 15
```

A frequência de Nyquist é:

```python
FN = FS / 2
```

Logo:

```text
FN = 100 kHz
```

A frequência de corte normalizada é:

```python
Wn = FC / FN
```

Esperado:

```text
Wn = 0.156
```

---

## Etapa 1.1 — Obter os coeficientes do Butterworth

Pode usar:

```python
from scipy.signal import butter

b, a = butter(
    ORDEM,
    FC,
    btype="low",
    fs=FS
)
```

### Importante

Explique que:

- `b` contém os coeficientes do numerador.
- `a` contém os coeficientes do denominador.
- Um filtro IIR possui realimentação.
- O valor atual da saída depende de entradas atuais/passadas e também de saídas anteriores.

Mostrar:

```python
print("b =", b)
print("a =", a)
```

---

# Etapa 1.2 — Implementar manualmente o filtro IIR

Evitar usar `lfilter()` inicialmente.

Implementar diretamente a equação de diferenças:

```text
y[n] =
b0*x[n]
+ b1*x[n-1]
+ ...
+ bM*x[n-M]

- a1*y[n-1]
- a2*y[n-2]
- ...
- aN*y[n-N]
```

Como normalmente `a[0] = 1`, podemos usar:

```python
def aplicar_filtro_iir(x, b, a):
    y = np.zeros(len(x))

    for n in range(len(x)):

        # Parte de avanço:
        # usa a amostra atual e as amostras passadas da entrada
        for k in range(len(b)):
            if n - k >= 0:
                y[n] += b[k] * x[n - k]

        # Parte de realimentação:
        # usa saídas anteriores do próprio filtro
        for k in range(1, len(a)):
            if n - k >= 0:
                y[n] -= a[k] * y[n - k]

        y[n] = y[n] / a[0]

    return y
```

Comentar cada parte.

### Observação importante

Filtros IIR de ordem alta podem apresentar problemas numéricos quando implementados diretamente dessa forma.

Para fins didáticos, primeiro tente esta implementação.

Se houver comportamento claramente instável ou valores numéricos absurdos, documentar isso e usar `scipy.signal.sosfilt` apenas como alternativa de estabilidade.

Não substituir silenciosamente a versão manual.

---

# Etapa 1.3 — Criar um sinal sintético

Criar um sinal contendo quatro senoides:

```text
5 kHz
10 kHz
30 kHz
60 kHz
```

As duas primeiras estão abaixo do corte de 15,6 kHz.

As duas últimas estão acima.

Sugestão:

```python
DURACAO = 0.01

t = np.arange(0, DURACAO, 1 / FS)

x = (
    np.sin(2*np.pi*5_000*t)
    + np.sin(2*np.pi*10_000*t)
    + np.sin(2*np.pi*30_000*t)
    + np.sin(2*np.pi*60_000*t)
)
```

Explicar que esse sinal artificial é útil porque sabemos exatamente quais frequências estão presentes.

---

# Etapa 1.4 — Filtrar o sinal

Usar a implementação manual:

```python
y = aplicar_filtro_iir(x, b, a)
```

Plotar:

1. sinal antes;
2. sinal depois.

Como o sinal temporal pode ficar visualmente confuso, mostrar apenas uma pequena janela.

Exemplo:

```python
plt.xlim(0, 0.001)
```

---

# Etapa 1.5 — Calcular o espectro com FFT

Pode usar:

```python
X = np.fft.fft(x)
Y = np.fft.fft(y)
```

Evitar funções prontas de espectro.

Criar o eixo de frequências manualmente:

```python
freq = np.fft.fftfreq(len(x), d=1/FS)
```

Como o sinal é real, inicialmente mostrar apenas as frequências positivas:

```python
mascara = freq >= 0
```

Magnitude:

```python
mag_X = np.abs(X)
mag_Y = np.abs(Y)
```

---

# Etapa 1.6 — Resposta em frequência do filtro

Não usar `freqz()` na primeira versão.

Calcular manualmente a resposta:

```text
         b0 + b1 z^-1 + ... + bM z^-M
H(z) = --------------------------------
         a0 + a1 z^-1 + ... + aN z^-N
```

Na circunferência unitária:

```text
z = exp(j*w)
```

Então:

```python
frequencias = np.linspace(0, FS/2, 2000)

H = []

for f in frequencias:
    w = 2 * np.pi * f / FS

    numerador = 0
    denominador = 0

    for k in range(len(b)):
        numerador += b[k] * np.exp(-1j * w * k)

    for k in range(len(a)):
        denominador += a[k] * np.exp(-1j * w * k)

    H.append(numerador / denominador)

H = np.array(H)
```

Converter para dB:

```python
H_db = 20 * np.log10(np.abs(H) + 1e-12)
```

O pequeno valor `1e-12` serve apenas para evitar `log(0)`.

---

# Etapa 1.7 — Verificar a frequência de corte

Encontrar o ponto mais próximo de 15,6 kHz:

```python
indice = np.argmin(np.abs(frequencias - FC))

print(
    "Ganho em 15,6 kHz:",
    H_db[indice],
    "dB"
)
```

Esperado:

```text
aproximadamente -3 dB
```

Essa será uma das principais validações numéricas do Experimento 1.

---

# Etapa 1.8 — Medir atenuação

Medir o ganho em:

```text
5 kHz
10 kHz
15,6 kHz
30 kHz
60 kHz
```

Criar uma pequena função:

```python
def ganho_em_frequencia(f_alvo, frequencias, H_db):
    indice = np.argmin(np.abs(frequencias - f_alvo))
    return H_db[indice]
```

Imprimir os resultados em texto simples.

Não é necessário usar pandas.

---

# Resultados esperados do Experimento 1

Gerar:

```text
resultado_filtro.png
espectro_antes_depois.png
```

E imprimir no terminal algo semelhante a:

```text
Ganho em 5 kHz: ...
Ganho em 10 kHz: ...
Ganho em 15.6 kHz: aproximadamente -3 dB
Ganho em 30 kHz: ...
Ganho em 60 kHz: ...
```

---

# Experimento 2 — Aplicação em sinais I/Q

## Objetivo

Usar uma amostra do RadioML2016.10a para demonstrar a ideia principal do artigo:

```text
perturbação com energia fora da banda
                ↓
          filtro passa-baixas
                ↓
perturbação concentrada na faixa desejada
```

Não implementar inicialmente:

- CNN;
- FGSM;
- PGD;
- MIM;
- FCIAA completo.

---

# Etapa 2.1 — Carregar uma única amostra

Escolher inicialmente:

```text
Modulação: QPSK
SNR: 18 dB
```

Carregar apenas uma janela de 128 amostras.

O formato deve ser aproximadamente:

```text
(2, 128)
```

onde:

```python
I = amostra[0]
Q = amostra[1]
```

Criar o sinal complexo:

```python
x = I + 1j * Q
```

Explicar:

```text
I = componente em fase
Q = componente em quadratura
```

---

# Etapa 2.2 — Visualizar I e Q

Fazer um gráfico simples:

```text
I[n]
Q[n]
```

Não adicionar muitas visualizações inicialmente.

---

# Etapa 2.3 — Espectro do sinal I/Q

Calcular:

```python
X = np.fft.fft(x)
freq = np.fft.fftfreq(len(x), d=1/FS)
```

Como o sinal é complexo, mostrar frequências negativas e positivas.

Usar:

```python
X_shift = np.fft.fftshift(X)
freq_shift = np.fft.fftshift(freq)
```

Plotar de:

```text
-100 kHz a +100 kHz
```

---

# Etapa 2.4 — Criar perturbação simples

Não tentar reproduzir um ataque adversarial real neste momento.

Criar apenas ruído complexo:

```python
np.random.seed(0)

delta = (
    np.random.normal(0, 1, len(x))
    + 1j * np.random.normal(0, 1, len(x))
)
```

Depois reduzir sua amplitude:

```python
delta = 0.05 * delta
```

O valor `0.05` é apenas inicial.

Deve ser declarado como uma suposição experimental.

---

# Etapa 2.5 — Filtrar a perturbação, não o sinal original

Este ponto é importante.

O artigo não usa o Butterworth apenas como um filtro receptor aplicado diretamente ao sinal original.

Nesta reprodução:

```python
delta_filtrada_real = aplicar_filtro_iir(
    delta.real, b, a
)

delta_filtrada_imag = aplicar_filtro_iir(
    delta.imag, b, a
)

delta_filtrada = (
    delta_filtrada_real
    + 1j * delta_filtrada_imag
)
```

Então criar:

```python
x_adv = x + delta

x_adv_filtrado = x + delta_filtrada
```

---

# Etapa 2.6 — Comparar os espectros

Gerar FFT de:

```text
delta
delta_filtrada
```

Plotar no mesmo gráfico ou em dois gráficos simples.

O objetivo visual é mostrar que a perturbação original possui energia espalhada pelo espectro enquanto a filtrada concentra energia principalmente abaixo de aproximadamente 15,6 kHz.

---

# Etapa 2.7 — Calcular SPR

Usar a fórmula do artigo:

```text
           energia do sinal original
SPR = 10 log10 -----------------------------
           energia da perturbação
```

Implementar:

```python
def calcular_spr(x, x_adv):

    energia_sinal = np.sum(
        np.abs(x) ** 2
    )

    energia_perturbacao = np.sum(
        np.abs(x - x_adv) ** 2
    )

    spr = 10 * np.log10(
        energia_sinal / energia_perturbacao
    )

    return spr
```

Calcular:

```python
spr_sem_filtro = calcular_spr(x, x_adv)

spr_com_filtro = calcular_spr(
    x,
    x_adv_filtrado
)
```

---

# Etapa 2.8 — Calcular OBER

Para manter simples, definir nesta reprodução:

```text
Dentro da banda:
-15,6 kHz <= f <= +15,6 kHz

Fora da banda:
|f| > 15,6 kHz
```

Implementar:

```python
def calcular_ober(delta, fs, fc):

    D = np.fft.fft(delta)

    freq = np.fft.fftfreq(
        len(delta),
        d=1/fs
    )

    potencia = np.abs(D) ** 2

    mascara_fora = np.abs(freq) > fc

    energia_total = np.sum(potencia)

    energia_fora = np.sum(
        potencia[mascara_fora]
    )

    ober_percentual = (
        100 * energia_fora / energia_total
    )

    return ober_percentual
```

Calcular:

```python
ober_sem_filtro = calcular_ober(
    delta,
    FS,
    FC
)

ober_com_filtro = calcular_ober(
    delta_filtrada,
    FS,
    FC
)
```

---

# Importante sobre OBER

O artigo possui uma inconsistência que deve ser comentada.

A equação apresentada usa:

```text
10 log10(Efora / Etotal)
```

mas os resultados da figura são apresentados em porcentagem.

Nesta reprodução:

- usar OBER percentual como principal;
- opcionalmente calcular também OBER em dB;
- explicar essa diferença no relatório.

Não tentar forçar o resultado para chegar em 1,53%.

---

# Resultado final do Experimento 2

Imprimir algo semelhante a:

```text
SPR sem filtro: ...
SPR com filtro: ...

OBER sem filtro: ... %
OBER com filtro: ... %

Valor reportado pelo FCIAA:
SPR = 16.90 dB
OBER = 1.53 %
```

Deixar explícito:

> Os nossos números não precisam coincidir com 16,90 dB e 1,53%, pois estamos usando uma perturbação sintética, enquanto o artigo utiliza uma perturbação produzida iterativamente pelo algoritmo FCIAA.

---

# Figuras mínimas

Produzir apenas estas inicialmente:

## Experimento 1

1. `resposta_frequencia.png`
2. `espectro_sintetico_antes_depois.png`

## Experimento 2

3. `iq_tempo.png`
4. `perturbacao_espectro_antes_depois.png`

Evitar gerar muitas figuras redundantes.

---

# Comentários obrigatórios no código

O agente deve escrever comentários que expliquem:

- o que é frequência de amostragem;
- o que é Nyquist;
- o que significa frequência de corte;
- por que Butterworth é passa-baixas;
- diferença entre FIR e IIR;
- por que um IIR possui realimentação;
- o significado de I e Q;
- o que a FFT mostra;
- o que significa energia fora da banda;
- significado de SPR;
- significado de OBER.

Os comentários devem ser simples e curtos.

Exemplo:

```python
# A frequência de Nyquist é metade da frequência de amostragem.
# Neste experimento:
# fs = 200 kHz -> Nyquist = 100 kHz
```

---

# O que evitar

Não fazer:

- orientação a objetos;
- classes;
- dataclasses;
- arquivos de configuração;
- logging complexo;
- CLI;
- argparse;
- notebooks enormes;
- frameworks;
- abstrações genéricas desnecessárias;
- otimizações prematuras.

Não transformar uma atividade didática em um software de produção.

---

# Regra para funções prontas

## Pode usar

```text
np.fft.fft
np.fft.fftfreq
np.fft.fftshift
np.abs
np.sum
np.log10
np.exp
np.sin
np.arange
np.linspace
np.convolve
scipy.signal.butter
matplotlib
```

## Evitar inicialmente

```text
scipy.signal.freqz
scipy.signal.lfilter
scipy.signal.filtfilt
scipy.signal.sosfreqz
scipy.signal.sosfilt
```

Essas funções podem ser usadas depois para comparar ou validar nossa implementação manual.

---

# Estratégia de validação

Depois que a versão manual estiver funcionando, é permitido comparar com funções prontas do SciPy.

Exemplo:

```text
resultado manual
versus
scipy.signal.lfilter
```

Se os resultados forem próximos, isso ajuda a validar a implementação.

Mas a explicação principal deve continuar baseada na versão manual.

---

# Ordem de implementação

Execute exatamente nesta ordem:

1. Criar constantes `FS`, `FC` e `ORDEM`.
2. Obter `b` e `a` com `scipy.signal.butter`.
3. Implementar `aplicar_filtro_iir()`.
4. Criar sinal sintético.
5. Filtrar sinal sintético.
6. Calcular FFT antes/depois.
7. Implementar resposta em frequência manual.
8. Confirmar aproximadamente -3 dB em 15,6 kHz.
9. Medir ganho em frequências escolhidas.
10. Salvar gráficos do Experimento 1.
11. Carregar uma amostra do RadioML.
12. Separar I e Q.
13. Criar sinal complexo.
14. Criar perturbação sintética.
15. Filtrar parte real e imaginária da perturbação.
16. Plotar espectro da perturbação antes/depois.
17. Implementar SPR.
18. Implementar OBER.
19. Comparar métricas antes/depois.
20. Salvar resultados.

Não avançar para CNN ou FCIAA completo antes de esses passos funcionarem.

---

# Forma de trabalhar com o usuário

Implemente uma etapa por vez.

Para cada etapa:

1. explique brevemente o conceito;
2. apresente o código;
3. diga o resultado esperado;
4. explique como verificar se está correto;
5. só então avance.

Se aparecer erro, investigar o erro atual em vez de reescrever todo o projeto.

O usuário é iniciante em PDS, mas já programa. Portanto:

- não explicar sintaxe Python básica demais;
- explicar conceitos de sinais;
- relacionar matemática com o código;
- não esconder etapas matemáticas atrás de funções prontas.

---

# Resultado esperado do projeto

Ao final, o usuário deve conseguir explicar:

1. O que é um filtro Butterworth.
2. Por que esse filtro é IIR.
3. O que significa ordem 15.
4. O que representa uma frequência de corte de 15,6 kHz.
5. O efeito do filtro no domínio da frequência.
6. Por que o artigo filtra a perturbação.
7. Como a filtragem altera a energia fora da banda.
8. O significado de SPR.
9. O significado de OBER.
10. Por que os resultados da reprodução simplificada não são idênticos aos valores do FCIAA completo.
