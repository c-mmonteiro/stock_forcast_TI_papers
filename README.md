Código utilizado para gerar os resultados do artigo da CSBC 2024.
https://sol.sbc.org.br/index.php/encompif/article/view/29292/29097

Utilize o Python 3.10 e limpe todas as bibliotecas. Não esqueça de verificar se está utilizando o pip do interpretador utilizado. Lembrar de colocar tanto o python 3.10 quanto o seu pip no PATH do windows.

Para utilizar estes scripts é necessário instalar os pacotes do requirements.txt e o ELM a partir do github:
https://github.com/5663015/elm

Para a instalação do elm é necessário alterar a linha 106 e 128 trocando o comando:

time.clock()

por 

time.process_time()

tanto no arquivo da pasta raiz, quanto no da pasta ./bild/lib e então proceder com o procedimento de instação sugerido.

