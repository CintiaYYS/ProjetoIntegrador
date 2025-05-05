import os
import pandas as pd
import random
import spacy
from spacy.lookups import load_lookups
from spacy.util import minibatch
from pathlib import Path as Path
from spacy.training import Example
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import swifter


# Carregar modelo de linguagem em português do spaCy
nlp = spacy.load("pt_core_news_lg")

#Caminho do arquivo csv que armazena as informações extraidas dos posts
caminho_base_posts = "dados/processados/base_posts.csv"

# Caminho onde a base_rotulada será criada e em seguida usada para treino do modelo
caminho_base_rotulada = "dados/processados/base_rotulada.csv"

# Caminho da base_rotulada usada para treinar o modelo para classificar o texto em Encontrado ou Desaparecido
caminho_base_rotulada_encontrado = "dados/processados/base_rotulada_encontrado.csv"

# Caminho do arquivo descartados.csv
caminho_descartados = "dados/processados/descartados.csv"

# Colunas que serão usadas
colunas = ['Content','Url']

# Caminho do modelo nlp
caminho_modelo_nlp = "dados/modelos/modelo_spacy"

caminho_modelo_nlp_encontrado = "dados/modelos/modelo_spacy_encontrado"

# Classe para transformar texto em embeddings com spaCy
class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load("pt_core_news_lg")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.nlp(text).vector for text in X]



# Função definida para unir vários arquivos em um só
def une_csv():
    """
    Carrega os arquivos csv
    Filtra apenas as colunas que serão utilizadas
    Remove registros duplicados (Url e Content iguais)
    Caso tenha registros com Url repetidas: Concatena Content e mantém apenas um registro em df_posts
    
    Quando Content é NaN, ainda é preciso analisar a URL, assim, o valor de URL passa a ser atribuido a Content
    Ao final, será salvo o arquivo base_posts.csv que contém apenas um registro por Url
    """

    caminho_csv = os.path.abspath("dados/brutos/csv")
    
    # Criando um arquivo de descarte vazio
    descartados = pd.DataFrame()
    descartados.to_csv(caminho_descartados)

    arquivos = [os.path.join(caminho_csv, f) for f in os.listdir(caminho_csv) if f.endswith('.csv')]

    df_unido = pd.concat([pd.read_csv(arquivo) for arquivo in arquivos], ignore_index=True)

    # Selecionando apenas as colunas de interesse
    df_posts = df_unido[colunas]


    # Quando a postagem tem Content == NaN, significa que a postagem foi removida, logo, o registro deve ser removido
    df_posts = df_posts.dropna()


    df_posts.loc[:, 'Url'] = df_posts['Url'].astype(str)
    df_posts.loc[:, 'Content'] = df_posts['Content'].astype(str)
    
    
    # Removendo registros totalmente ['Url','Content'] iguais
    df_posts = df_posts.drop_duplicates(subset=colunas,keep='first')
    
    # Verificando os registros que tem Url iguais mas Content diferentes
    # Precisa concatenar Content pois pode conter informações adicionais
    df_url_repetidas = df_posts[df_posts.duplicated(subset='Url', keep='first')]
    
    
    if df_url_repetidas.shape[0] > 0:
        # Removendo registros com Url iguais e Content diferentes    
        df_posts = remover_registros_duplicados(df_posts,df_url_repetidas, motivo='URL repetida e Content diferente')   
        
    # Salvando o arquivo com os posts que serão analisados
    df_posts.to_csv(caminho_base_posts, index=False)

    

    




def remover_registros_duplicados(df_posts_entrada, df_duplicados, motivo):
    """
    Remove registros duplicados com base na coluna 'Url', mantendo um representante e
    concatenando os valores de 'Content'. Adiciona os registros removidos ao DataFrame de descarte.

    Parâmetros:
    - df_posts_entrada: DataFrame original com os posts
    - df_duplicados: subconjunto com URLs duplicadas
    - motivo: string com o motivo do descarte

    Retorna:
    - df_dados_atualizado: DataFrame sem duplicações
    """
    df_posts_entrada = df_posts_entrada.copy()
    df_duplicados = df_duplicados.copy()
  


    df_unificado = df_duplicados.groupby('Url', as_index=False).agg({
        'Content': lambda x: ' '.join(x.dropna().unique())
    })

    df_aux = df_duplicados.merge(df_unificado, on='Url', how='left', suffixes=('', '_agrupado'))


    df_aux['Content'] = df_aux['Content'].dropna()
    df_aux['Content_agrupado'] = df_aux['Content_agrupado'].dropna()
    df_descartar = df_aux[df_aux['Content'] != df_aux['Content_agrupado']].copy()
    df_descartar['Motivo'] = motivo

    df_dados_atualizado = df_posts_entrada[~df_posts_entrada['Url'].isin(df_duplicados['Url'])]
    df_dados_atualizado = pd.concat([df_dados_atualizado, df_unificado], ignore_index=True)

    atualiza_descartado(df_descartar)


    return df_dados_atualizado



def remover_registros(df_entrada, df_remover, motivo):
    """
    Remove de df_entrada os registros que estão em df_remover
    Adiciona esses registros ao df_descartados com uma nova coluna 'motivo'
    
    Retorna:
        df_entrada_atualizado
    """
    # Garantir que temos cópias para evitar alterações indesejadas
    df_remover = df_remover.copy()

    df_entrada = df_entrada.copy()

    
    # Remover do df_entrada os registros que estão no df_remover (com base no índice ou colunas específicas)
    # Aqui usaremos a interseção das linhas completas (mais seguro se os DataFrames forem iguais em estrutura)
    df_entrada_atualizado = df_entrada.merge(df_remover, 
                                             how='outer', 
                                             indicator=True)
    df_entrada_atualizado = df_entrada_atualizado[df_entrada_atualizado['_merge'] == 'left_only']
    df_entrada_atualizado = df_entrada_atualizado.drop(columns=['_merge'])
    
    df_remover['Motivo'] = motivo
    
    atualiza_descartado(df_remover)

    return df_entrada_atualizado



def atualiza_descartado(df_remover):
    """
    Os registros em descartados só são atualizados quando há uma remoção
    E são utilizados apenas para conferência
    """
    df_descartado = pd.read_csv(caminho_descartados)
    df_descartado = pd.concat([df_remover,df_descartado], ignore_index=True)
    df_descartado.to_csv(caminho_descartados,index=False)



def unifica_registros_por_imagem(imagem_mantida, imagem_removida):
    
    # Verifica se o CSV existe
    if not Path(caminho_base_posts).exists():
        print(f"Arquivo CSV não encontrado: {caminho_base_posts}")
        return

    # Carrega o CSV
    df = pd.read_csv(caminho_base_posts)
    
    # Verifica se ambas as imagens existem no CSV
    if imagem_mantida not in df["nome_imagem"].values or imagem_removida not in df["nome_imagem"].values:
        print("Uma ou ambas as imagens não estão no CSV.")
        return

    # Recupera os registros
    registro_mantido = df[df["nome_imagem"] == imagem_mantida].iloc[0]
    registro_removido = df[df["nome_imagem"] == imagem_removida].iloc[0]

    # Unifica as características e o rótulo (evita duplicar informações)
    caracteristicas_mantidas = str(registro_mantido.get("Content", ""))
    caracteristicas_removidas = str(registro_removido.get("Content", ""))
    novas_caracteristicas = "; ".join(filter(None, {caracteristicas_mantidas, caracteristicas_removidas}))
    if classificar_texto(novas_caracteristicas) == 'relacionado':    
        # Atualiza o registro mantido
        df.loc[df["nome_imagem"] == imagem_mantida, "Content"] = novas_caracteristicas
    else:
        # Um dos textos é nao-relacionado, assim, precisa ser removido da base_posts    
        df = df[df["nome_imagem"] != imagem_mantida]

    # Remove o registro 
    df = df[df["nome_imagem"] != imagem_removida]


    # Salva o CSV atualizado
    df.to_csv(caminho_base_posts, index=False)


def ETL_posts():
    """
    Função que reduz ao máximo os registros a serem analisados
    Mais para frente, as imagens deverão ser analisadas e quanto menor o número de imagens
    mais rápido será o processamento.
    Assim, é necessário fazer uma limpeza utilizando os textos.

    Neste ponto, o arquivo csv já foi gerado com os registros duplicados removido e 
    registros que possuem mesma URL mas Content diferentes foram unificados

    A função ELT_posts deve:
        1- Remover registros NaN
        2 -Remover registros que estejam relacionados com outros animais como: gato, calopsita, ...
        3 -Remover registros relacionados a adotação e resgate
    """
    


    try:
        
        df = pd.read_csv(caminho_base_posts)
        criar_modelo_pln()
        df["relacionado"] = df["Content"].swifter.apply(classificar_texto)
        df_filtrado = df[df["relacionado"] == 'relacionado'].reset_index(drop=True)
        df_filtrado.to_csv(caminho_base_posts,index=False)

    except:
        une_csv()
        df = pd.read_csv(caminho_base_posts)
        ETL_posts()
    






def cria_base_rotulada_texto():
    """
    Cria uma base csv que será utilizada para treinar o modelos de Procecssamento de Linguagem Natural
    """

    # Caminhos dos arquivos
    caminho_relacionado = "dados/brutos/txt/relacionado.txt"
    caminho_nao_relacionado = "dados/brutos/txt/nao-relacionado.txt"
    
    

    # Lê os arquivos como listas de frases
    with open(caminho_relacionado, encoding='utf-8') as f:
        relacionados = [linha.strip() for linha in f if linha.strip()]

    with open(caminho_nao_relacionado, encoding='utf-8') as f:
        nao_relacionados = [linha.strip() for linha in f if linha.strip()]

    # Monta o DataFrame
    textos = relacionados + nao_relacionados
    rotulos = ['relacionado'] * len(relacionados) + ['nao-relacionado'] * len(nao_relacionados)

    df = pd.DataFrame({
        'texto': textos,
        'rotulo': rotulos
    })

    df.to_csv(caminho_base_rotulada, index=False)






def classificar_texto(texto):
    """
    Classifica o texto passado como parâmetro
    Como o modelo gera uma pontuação
    é necessário verificar se a pontuação do rótulo 'relacionado' é maior que o 'nao-relacionado'
    
    Retorna True se a pontuação de 'relacionado' for maior que a de 'nao-relacionado'
    Retorna False se a pontuação de 'nao-relacionado' for maior que a de 'relacionado'
    """
    nlp = spacy.load(caminho_modelo_nlp)

    doc = nlp(str(texto))  # garante que seja string
    scores = doc.cats       # retorna um dicionário {'relacionado': 0.9, 'nao-relacionado': 0.1}
    return "relacionado" if (scores['relacionado'] > scores['nao-relacionado']) else 'nao-relacionado'





def criar_modelo_pln():

    # Se o diretório já existe, significa que o modelo já foi treinado e criado
    if os.path.isdir(caminho_modelo_nlp):
        return 
    else:
        print("Iniciando treinamento Relacionado/Nao-Relacionado")
        cria_base_rotulada_texto()

       
        nlp = spacy.load('pt_core_news_lg')

        df = pd.read_csv(caminho_base_rotulada)

        # Misturando os registro em df
        df = df.sample(frac=1).reset_index(drop=True)

        # Adiciona o componente de classificação de texto
        if 'textcat' not in nlp.pipe_names:
            textcat = nlp.add_pipe('textcat', last=True)
        else:
            textcat = nlp.get_pipe('textcat')

        # Define as categorias

        textcat.add_label('relacionado')
        textcat.add_label('nao-relacionado')

        exemplos_treino = []
        for texto, rotulo in zip(df['texto'], df['rotulo']):
            doc = nlp.make_doc(texto)
            anotacao = {
                'cats': {
                    'relacionado': 1 if rotulo == 'relacionado' else 0,
                    'nao-relacionado': 1 if rotulo == 'nao-relacionado' else 0
                }
            }

        exemplos_treino.append(Example.from_dict(doc, anotacao))

        # Treinamento
        
        optimizer = nlp.initialize()
        for epoca in range(10):
            random.shuffle(exemplos_treino)
            lotes = minibatch(exemplos_treino, size=8)
            for lote in lotes:
                lote_dados = [(ex.text, ex.reference) for ex in lote]
                textos, anotacoes = zip(*lote_dados)
                nlp.update(lote, sgd=optimizer)
                
        # Salva o modelo
        nlp.to_disk(caminho_modelo_nlp)




def classificar_texto_encontrado(texto):
    """
    Classifica o texto passado como parâmetro
    Como o modelo gera uma pontuação
    é necessário verificar se a pontuação do rótulo 'relacionado' é maior que o 'nao-relacionado'
    
    Retorna True se a pontuação de 'relacionado' for maior que a de 'nao-relacionado'
    Retorna False se a pontuação de 'nao-relacionado' for maior que a de 'relacionado'
    """
    nlp = spacy.load(caminho_modelo_nlp_encontrado)

    doc = nlp(str(texto))  # garante que seja string
    scores = doc.cats       # retorna um dicionário {'relacionado': 0.9, 'nao-relacionado': 0.1}
    return "encontrado" if (scores['encontrado'] > scores['desaparecido']) else 'desaparecido'




def classifica_encontrado_desaparecido():

    df = pd.read_csv(caminho_base_posts)
    
    criar_modelo_pln_encontrado()

    # Cada post passa a ter um status: encontrado ou desaparecido
    df["status"] = df["Content"].swifter.apply(classificar_texto_encontrado)
    
    df.to_csv(caminho_base_posts,index=False)
    




def cria_base_rotulada_encontrado_desaparecido():
    """
    Cria uma base csv que será utilizada para treinar o modelos de Procecssamento de Linguagem Natural
    """

    # Caminhos dos arquivos
    caminho_relacionado = "dados/brutos/txt/encontrado.txt"
    caminho_nao_relacionado = "dados/brutos/txt/desaparecido.txt"
    
    

    # Lê os arquivos como listas de frases
    with open(caminho_relacionado, encoding='utf-8') as f:
        relacionados = [linha.strip() for linha in f if linha.strip()]

    with open(caminho_nao_relacionado, encoding='utf-8') as f:
        nao_relacionados = [linha.strip() for linha in f if linha.strip()]

    # Monta o DataFrame
    textos = relacionados + nao_relacionados
    rotulos = ['encontrado'] * len(relacionados) + ['desaparecido'] * len(nao_relacionados)

    df = pd.DataFrame({
        'texto': textos,
        'rotulo': rotulos
    })

    df.to_csv(caminho_base_rotulada_encontrado, index=False)




def criar_modelo_pln_encontrado():

    
    # Se o diretório já existe, significa que o modelo já foi treinado e criado
    if os.path.isdir(caminho_modelo_nlp_encontrado):
        return 
    
    else:
        cria_base_rotulada_encontrado_desaparecido()
        print("Iniciando treinamento modelo Desaparecido/Encontrado")
        # Carrega o modelo spaCy em português
        nlp = spacy.load("pt_core_news_lg")

        df = pd.read_csv(caminho_base_rotulada_encontrado)

        # Misturando os registro em df
        df = df.sample(frac=1).reset_index(drop=True)

        # Adiciona o componente de classificação de texto
        if 'textcat' not in nlp.pipe_names:
            textcat = nlp.add_pipe('textcat', last=True)
        else:
            textcat = nlp.get_pipe('textcat')

        # Define as categorias
        
        textcat.add_label('encontrado')
        textcat.add_label('desaparecido')

        exemplos_treino = []
        for texto, rotulo in zip(df['texto'], df['rotulo']):
            doc = nlp.make_doc(texto)
            anotacao = {
                'cats': {
                    'encontrado': 1 if rotulo == 'encontrado' else 0,
                    'desaparecido': 1 if rotulo == 'desaparecido' else 0
                }
            }

        exemplos_treino.append(Example.from_dict(doc, anotacao))

        # Treinamento
        
        optimizer = nlp.initialize()
        for epoca in range(10):
            random.shuffle(exemplos_treino)
            lotes = minibatch(exemplos_treino, size=8)
            for lote in lotes:
                lote_dados = [(ex.text, ex.reference) for ex in lote]
                textos, anotacoes = zip(*lote_dados)
                nlp.update(lote, sgd=optimizer)
                
        # Salva o modelo
        nlp.to_disk(caminho_modelo_nlp_encontrado)

