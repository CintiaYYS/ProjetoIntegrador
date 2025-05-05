from codigo_fonte.processamento.csv_utilidades import  ETL_posts, classifica_encontrado_desaparecido
from codigo_fonte.processamento.imagens_utilidades import  comparar_todas_imagens_recorte, executar
from codigo_fonte.scrapping.facebook_scraper_playwright import baixarImagens

if __name__ == "__main__":

        print("Iniciando sistema")
        ETL_posts()
        baixarImagens()
        comparar_todas_imagens_recorte()
        classifica_encontrado_desaparecido()
        print("Iniciando busca")        
        executar()
