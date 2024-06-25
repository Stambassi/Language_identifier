
#AUGUSTO 2019
#Proximos Passos:
# 1. Base de dados a partir de : https://1000mostcommonwords.com/1000-most-common-portuguese-words/
# 2. Ler arquivo em Pyhton: https://realpython.com/python-csv/#reading-csv-files-with-csv
# 3. Fazer analise de erro: % de acerto com relação ao numero de iterações
# 4. Fazer duas camadas com Pytorch: https://medium.com/coinmonks/create-a-neural-network-in-pytorch-and-make-your-life-simpler-ec5367895199
# 5. Fazer outra lingua
import numpy
import csv


alphabet = ["-","’","a","á","à","ã","â","b","c","ç","d","e","é","è","ê", "f", "g", "h","i","í", "j","k","l","m", "n","o", "õ","ó","ô","p", "q","r","s","t","u","ú","ù","û","v","w","x","y","z"]



def read_files():
    global words, labels
    print("Lendo arquivo...")
    words=''
    labels = []
    arq = ""

    arq = "base_pt_en.csv"

    with open(arq) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print("Primeira linha...")
                line_count += 1
            else:
                words += row[0] + ' '
                labels.append(float(row[1]))
                line_count += 1

        print(f'Processed {line_count} lines.')
    print("Palavras:")
    print(words)
    print("Rotulos:")
    print(labels)

read_files()

splited_words=words.split()
print("aqui")
print(splited_words)

#Encontra o maximo de letras nas palavras se for maior que 50:
max_word_length = 0
def find_max_number_letters():
    global i, max_word_length
    for i in range(len(splited_words)):
        if len(splited_words[i]) > max_word_length:
            max_word_length = len(splited_words[i])
    print("Max word length ")
    print(max_word_length)
max_word_length = 50

find_max_number_letters()

def one_hot_encoding(words):
    # Creates empty lines for the encoding
    empty_line = []
    for i in range(len(alphabet)):
        empty_line.append(0)
    global TsIn
    # Converts the words to one-hot-encoding notation
    num_words = len(words.split())
    wordlist = list(words)
    wordlist.append(' ')
    space = wordlist.index(' ')
    TsIn = []
    # Creating the Matrix of matrix(TsIn)
    for i in range(num_words):
        matrix = []
        for e in range(space):
            line = []
            achou = 0
            for x in range(len(alphabet)):
                if wordlist[e] == alphabet[x]:
                    line.append(1)
                    achou = 1
                else:
                    line.append(0)
            if achou == 0:
                print("Caracter desconhecido")
                print(wordlist[e])
            matrix.append(line)
        for t in range(len(matrix), max_word_length):
            matrix.append(empty_line)

        TsIn.append(matrix)
        for d in range(0, space + 1):
            if len(wordlist) > 1:
                del wordlist[0]
            space = wordlist.index(' ')
    #print("Matriz 3D em lista:")
    #print(TsIn)
    temp = []
    for i in range(len(TsIn)):
        line = TsIn[i]
        temp2 = []
        for j in range(len(line)):
            temp2.extend(line[j])

        temp.append(temp2)
    TsIn = temp
    #print("Matriz 2D em lista:")
    #print(TsIn)
    TsIn = np.array(TsIn)
    #print("Matriz 2D em array:")
    #print(TsIn)


tsOut = np.array([labels]).T
sWeights = []


def training(trainSize):
    global sWeights, out
    np.random.seed(1)
    sWeights = 2 * np.random.random((len(alphabet) * max_word_length, 1)) - 1
    # distrubuicao uniforme entre 1, -1
    # [a,b),b<a : a+ b-a * randomSample + a
    # -1-1 * (
    for iteration in range(trainSize):
        if iteration%10 == 0:
            print(iteration)
        out = 1 / (1 + np.exp(-(np.dot(TsIn, sWeights))))
        error_derivative = (tsOut - out) * out * (1 - out)
        sWeights += np.dot(TsIn.T, error_derivative)


def testing():
    print("** Testando novos valores: **")

        #new_words =("cabeceira mamãe mommy canela panela caçarola mamão maple walking andança preto black stay morning sabedoria english katchup red troglodita yellow cafeína batata day morning dance limão irmã dança have done however mesa grace andarilho yes nada dancing")
        #right_answers = ['P','P', 'I', 'P','P','P','P','I','I','P','P','I','I','I','P','I','I','I', 'P', 'I', 'P', 'P', 'I', 'I', 'I', 'P', 'P', 'P', 'I','I','I','P','I','P','I','P','I']
        #new_words =("aeroporto assistência bigodes chá charuto chaveiro chuteira chuva concordar contagioso cordão divã encerramento esqui explosão falsificação milionário parafuso pesadelo principal rocha sinfonia sociedade víbora, vídeo, zoológico ")
        #right_answers = ['P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P']
    new_words =("computer")
    #right_answers = ['I']

    one_hot_encoding(new_words)
    out = 1 / (1 + np.exp(-(np.dot(TsIn, sWeights))))
    #print("Matriz de respostas: ")

    print(new_words)
    #print("Resposta certa: ")
    #print(right_answers)
    #final=[]
    #rigths=0
    #wrongs=0
    for i in range(len(out)):

        if out[i] > 0.5:
             print("Português")
             print(out)
        else:
             print("Inglês")
             print(out)

       # if final[i] == right_answers[i]:
           # rigths += 1
       # else:
           # wrongs += 1
            #print("Errou %d"%(i+1))
    #print("Previsoes: ")
    #print(final)
    #accuracy = rigths/len(out)*100
    #print("Acertos: %4.1f %% "%accuracy)


def all():
    train_size = 200
    for t in range(0, train_size, 10):
        one_hot_encoding(words)
        training(t)
        #print("Pesos depois do treino: ")
        #print(sWeights)
        testing()

all()

print("SWeights:")
print(sWeights)
