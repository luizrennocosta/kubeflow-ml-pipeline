# Introdução

Este projeto envolve a definição de um pipeline de Machine Learning de ponta a ponta que encobre principalmente os pontos de:

1. Ingestão de dados
2. Préprocessamento
    1. Limpeza do texto
    2. Tokenização
    3. Vetorização
    4. __Embedding__
3. Criação de conjunto de Treino/Validação/Teste (98%/1%/1%)
4. Treinamento de Modelo
    - Regressão logística
    - Deep model
        1. Embedding Layer
        2. Conv1D
        3. Global Max Pooling 1D
        4. Dense (MLP Fully connected)
        5. Dropout
5. Avaliação de Modelo
    - Matriz de Confusão
    - _Accuracy_
    - _Recall_
    - _Precision_
6. Deploy (\*)
    - SeldonCore

(\*): Devido a limitações de tempo, conhecimento da ferramenta e infraestrutura on-prem, esta etapa não foi totalmente concluída, apenas arquitetada em código.

# Ingestão
A fase de ingestão é muito simples, é apenas um request para baixar o arquivo especificado pelo desafio. Depois o arquivo é salvo fisicamente em um volume que é compartilhado com os próximos componentes.

# Préprocessamento
A parte de préprocessamento pode ser dividida em 4 estágios que são normalmente usadas para tratamento de datasets com o foco em _Natural Language Processing_. 
## 1- Limpeza
O passo de limpeza do texto remove alguns caracteres especiais, números, converte o texto para _lower case_ e remove _stopwords_. As _stopwords_ (como _the_, _where_, etc) são palavras comuns que podem afetar de maneira negativa algoritmos clássicos (como TF-IDF) devido à sua maior ocorrência em textos. Ex:

```python
clean('I Love Nachos!!') -> 'love nachos'
```

## 2- Tokenização
Essa etapa envolve trasnformar uma frase em uma lista de palavras, normalmente não encodadas. Ex:

```python
tokenize('love nachos') -> ['love', 'nachos']
```

## 3- Vetorização
Esse componente é responsavel pela criação de um vetor numérico que representa o documento. Neste caso, o componente cria um identificador único para cada palavra e depois converte a frase para estes identiicadores numéricos. Ex:

```python
vectorize(['love', 'nachos', 'love', 'guacamole']) -> [25, 39, 25, 13]
```

## 4- Embedding
Esta transformação converte o vetor numérico de identificadores únicos para um vetor contínuo de N dimensões baseado em algum modelo (como tf-idf, glove, word2vec, etc). No caso desse desafio, foi testado tanto o modelo tf-idf quando o glove para a _layer_ de _embedding_.

```python
embed([20, 39, 25, 13], N=5) -> [1.23, 0.38, 0.99, -1.2, -0.8]
```

# Modelos utilizados

Dois modelos foram utilzados, um de Regressão Logística utilizando uma transformação tf-idf para conversão do corpo de texto para vetores numéricos densos, e outro baseado em Redes Convolucionais de 1 dimesão com uma _layer_ de _embedding_ pré-treinada no Glove Dataset de 300 dimensões.

## Regressão Logística
O modelo de RL é um dos mais clássicos utilizados nos mais variados problemas de classificação. Ele se baseia no modelo de regressão linear utilizando uma função de ativação sigmoidal (função logística):

![lr_equation](http://www.sciweavers.org/upload/Tex2Img_1604552726/render.png)

onde `X` é o vetor de entrada, e `theta` a matriz de parâmetros a serem otimizados. Essa otimização de parâmetros é normalmente feita através de algoritmos como a Descida de Gradiente, que se utiliza de alguma função de _performance_ e através de pequenos incrementos ou decrementos nestes parâmetros, estima o gradiente e toma decisões de atualização desses parâmetros para a minimização ou maximação dessa função de _performance_. 
A regressão logística normalmente é resolvida maximizando a _likelihood_ ou minimizando uma perda de entropia cruzada. No caso de regressão logística binária, é possível encontrar uma atualização através do método de mínimos quadrados.

## Rede Neural Convolucional
Redes neurais tem como base o cálculo do gradiente do erro entre o valor predito pelo modelo e o valor esperado, normalmente este tipo de modelo realiza uma operação não linear entre a entrada e a saída se utilizando de diversas transformações lineares (operações matriciais) seguidas de ativações não lineares:

![nn_equation](http://www.sciweavers.org/upload/Tex2Img_1604553541/render.png)

Onde `f()` representa uma função de ativação não linear (normalmente "reLU", "tanh" ou "sigmoid"), e `W` o conjunto de parâmetros a serem otimizados. Redes neurais convolucionais não realizam uma operação de multiplicação de matrizes entre `W` e `X`, mas sim uma operação de **convolução**. Esse tipo de rede ficou muito famoso nos últimos anos pela sua capacidade de aprendizado (dado um conjunto suficientemente grande de dados) e aplicação para problemas de imagem, principalmente classificação.

O grande ponto positivo desse tipo de abordagem profunda é a capacidade de aprendizado "espacial" no conjunto de dados. Dependendo de como o conjunto é estruturado tensorialmente, as convoluções da rede conseguem encontrar os melhores filtros espaciais e extrair informações locais. Em outras palavras, no contexto de imagens, é possível que o modelo consiga identificar sessões espaciais da imagem e transformar essas partes em informação relevante (similar ao olho humano). No caso de texto, é possível traduzir isso para informações de contexto tanto intra quanto inter-frases.

O modelo escolhido para esse *approach* foi um super simples apenas para uma prova de conceito, envolvendo uma camada de *embedding* pré-treinada saindo em uma camada convolucional de 1 dimensão, seguido de uma camada de *pooling* e finalmente uma *layer* densa para a classificação supervisionada.

# Resultados

As análises de resultado dos modelos 