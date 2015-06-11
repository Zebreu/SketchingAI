import os
import random

from PIL import Image
from PIL import ImageDraw

import numpy
import matplotlib.pyplot as pyplot

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

caffe_root = os.path.abspath("/home/zebreu/Things/caffe-master")
dataroot = os.path.abspath("/home/zebreu/Downloads")
labelspath = os.path.abspath("/home/zebreu/Things/caffe-master/data/ilsvrc12/synset_words.txt")

featureroot = os.path.abspath("/home/zebreu/Things/sketchfeatures/")

import caffe

model_file = caffe_root+"/models/bvlc_reference_caffenet/deploy.prototxt"
pretrained = caffe_root+"/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"

saveddrawings = os.path.abspath("/media/zebreu/SavedDrawings")

####### Genetic parameters #######

iterations = 200
population_size = 200
crossover_probability = 0.8
mutation_probability = 0.2
large_mutation_probability = 0.2

tournament_size = 9
tournament_probability = 0.9

draw_settings = "selfchoice"

number_of_strokes = 14

if draw_settings == "selfchoice":
    number_of_parameters = 5
else:
    number_of_parameters = 4

number_genes = number_of_strokes*number_of_parameters



##########################

def test_old_old():
    with open(labelspath,"r") as opened_file:
        labels = opened_file.readlines()

    caffe.set_mode_cpu()

    net = caffe.Classifier(model_file, pretrained, 
                           mean=numpy.load(caffe_root+"/python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))

    test_image = dataroot+"/homecat.jpg"
    test_image1 = dataroot+"/241.png"

    input_image = caffe.io.load_image(test_image)
    input_image1 = caffe.io.load_image(test_image1)

    print net.blobs["fc6"].data.shape
    print numpy.max(net.blobs["fc6"].data)

    prediction = net.predict([input_image1])

    print net.blobs["fc6"].data.shape
    print numpy.max(net.blobs["fc6"].data)

    prediction = net.predict([input_image],oversample=False)

    print net.blobs["fc6"].data.shape
    print numpy.max(net.blobs["fc6"].data)
    
    print net.blobs["fc6"].data[1]
    return

    indices = numpy.argpartition(prediction[0],-10)[-10:]

    print prediction[0].argmax(), labels[prediction[0].argmax()]

    for index in indices:
        print labels[index]

def test_old():
    with open(labelspath,"r") as opened_file:
        labels = opened_file.readlines()

    caffe.set_mode_gpu()
    net = caffe.Net(model_file, pretrained, caffe.TEST)

    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    transformer.set_transpose("data",(2,0,1))
    transformer.set_mean("data",numpy.load(caffe_root+"/python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1))
    transformer.set_raw_scale("data",255)
    transformer.set_channel_swap("data",(2,1,0))

    net.blobs["data"].reshape(1,3,227,227)


    test_image = dataroot+"/homecat.jpg"
    test_image1 = dataroot+"/241.png"

    net.blobs["data"].data[...] = transformer.preprocess("data", caffe.io.load_image(test_image1))
    
    out = net.forward()    
    print net.blobs["fc6"].data.shape

    prediction = out["prob"]

    indices = numpy.argpartition(prediction[0],-10)[-10:]

    print prediction[0].argmax(), labels[prediction[0].argmax()]

    
    net.blobs["data"].data[...] = transformer.preprocess("data", caffe.io.load_image(test_image))

    out = net.forward()    
    print net.blobs["fc6"].data.shape
    

    prediction = out["prob"]

    indices = numpy.argpartition(prediction[0],-10)[-10:]

    print prediction[0].argmax(), labels[prediction[0].argmax()]

    for index in indices:
        print labels[index]

def test():
    with open(labelspath,"r") as opened_file:
        labels = opened_file.readlines()

    caffe.set_mode_gpu()
    net = caffe.Net(model_file, pretrained, caffe.TEST)

    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    transformer.set_transpose("data",(2,0,1))
    transformer.set_mean("data",numpy.load(caffe_root+"/python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1))
    transformer.set_raw_scale("data",255)
    transformer.set_channel_swap("data",(2,1,0))

    net.blobs["data"].reshape(1,3,227,227)

    test_image = os.path.join(saveddrawings,"zebra","190.png")

    net.blobs["data"].data[...] = transformer.preprocess("data", caffe.io.load_image(test_image))
    
    out = net.forward()    
    vector = net.blobs["fc6"].data.flatten()

    classifier = joblib.load(os.path.abspath("/home/zebreu/Things/sketchfeatures/classifier1.pk1"))

    print classifier.predict(vector)

    prediction = out["prob"]

    indices = numpy.argpartition(prediction[0],-10)[-10:]

    print prediction[0].argmax(), labels[prediction[0].argmax()]

    for index in indices:
        print labels[index]

def build_transformer(net):
    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    transformer.set_transpose("data",(2,0,1))
    transformer.set_mean("data",numpy.load(caffe_root+"/python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1))
    transformer.set_raw_scale("data",255)
    transformer.set_channel_swap("data",(2,1,0))

    return transformer

def preprocess():
    caffe.set_mode_gpu()
    net = caffe.Net(model_file, pretrained, caffe.TEST)

    transformer = build_transformer(net)    

    master_list5 = []
    master_list6 = []
    master_list7 = []
    master_list_labels = []
    datasetpath = os.path.abspath("/home/zebreu/Downloads/sketchdataset")
    for directory in sorted(os.listdir(datasetpath)):
        print directory
        for imagefile in os.listdir(os.path.join(datasetpath,directory)):
            imagepath = os.path.join(datasetpath,directory,imagefile)
            net.blobs["data"].reshape(1,3,227,227)
            net.blobs["data"].data[...] = transformer.preprocess("data", caffe.io.load_image(imagepath))
            net.forward(end="fc7")
            master_list5.append(net.blobs["pool5"].data.flatten())
            master_list6.append(net.blobs["fc6"].data.flatten())
            master_list7.append(net.blobs["fc7"].data.flatten())
            master_list_labels.append(directory)

    numpy.save("pool5features",numpy.array(master_list5))
    numpy.save("fc6features",numpy.array(master_list6))
    numpy.save("fc7features",numpy.array(master_list7))
    numpy.save("labels",numpy.array(master_list_labels))


def train():
    data = numpy.load(os.path.join(featureroot,"fc6features.npy"))
    #data = numpy.load(os.path.join(featureroot,"pool5features.npy"))
    labels = numpy.load(os.path.join(featureroot,"labels.npy"))

    labelslist = sorted(list(set(labels)))

    #print labelslist

    counter = 0.0
    for label in labelslist:
        labels[labels == label] = counter
        counter += 1

    labels = numpy.reshape(labels,(20000,1)).astype(numpy.float32)
    
    alldata = numpy.hstack((data,labels))

    numpy.random.shuffle(alldata)

    data = alldata[:,0:-1]
    labels = alldata[:,-1].astype(numpy.int)
    # 0.00001 seems best for linear SVC (both)
    for c in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        #classifier = LinearSVC(C=c)
        classifier = LogisticRegression(C=c)
        #classifier = SVC(C=c,kernel="linear")

        classifier.fit(data[0:10000], labels[0:10000])
        print classifier.score(data[10000:], labels[10000:]), c

    #joblib.dump(classifier,os.path.join(featureroot,"classifier1.pk1"))

def quick_train():
    data = numpy.load(os.path.join(featureroot,"fc6features.npy"))
    #data = numpy.load(os.path.join(featureroot,"pool5features.npy"))
    labels = numpy.load(os.path.join(featureroot,"labels.npy"))

    labelslist = sorted(list(set(labels)))

    #print labelslist

    counter = 0.0
    for label in labelslist:
        labels[labels == label] = counter
        counter += 1

    labels = numpy.reshape(labels,(20000,1)).astype(numpy.float32)
    
    alldata = numpy.hstack((data,labels))

    numpy.random.shuffle(alldata)

    data = alldata[:,0:-1]
    labels = alldata[:,-1].astype(numpy.int)

    #classifier = LinearSVC()
    classifier = SVC(C=0.00001,kernel="linear",probability=True)

    classifier.fit(data[0:2000], labels[0:2000])
    print classifier.score(data[2000:], labels[2000:])

    joblib.dump(classifier,os.path.join(featureroot,"classifierQuick.pk1"))

def min_or_max(scores):
    return max(scores)

def tournament(population, scores):
    chosen = random.sample(scores, tournament_size)
    if random.random() < tournament_probability:
        index = min_or_max(chosen)
        winner = population[index[1]]
        return winner
    else:
        return population[random.choice(chosen)[1]]

def selection(population, scores):
    selected = [tournament(population, scores) for _ in range(population_size)]
    return selected

def draw_individual(individual,iteration,desired_text):
    im = Image.new("L", (256,256),(256))
    draw = ImageDraw.Draw(im)
    index = 0

    if draw_settings == "straight":
        while index < number_genes-(number_of_parameters-1):
            draw.line(individual[index:index+number_of_parameters],fill=0, width=3) # width=12 is supersampling
            index += number_of_parameters
    elif draw_settings == "ellipse":
        while index < number_genes-(number_of_parameters-1):
            current = individual[index:index+number_of_parameters]
            if current[0] > current[2]:
                temp = current[0]
                current[0] = current[2]
                current[2] = temp
            if current[1] > current[3]:
                temp = current[1]
                current[1] = current[3]
                current[3] = temp

            draw.ellipse(current,outline=0) # not setting fill makes it transparent
            index += number_of_parameters
    
    elif draw_settings == "strellipse":
        swap = False
        while index < number_genes-(number_of_parameters-1):
            current = individual[index:index+number_of_parameters]
            if swap:
                if current[0] > current[2]:
                    temp = current[0]
                    current[0] = current[2]
                    current[2] = temp
                if current[1] > current[3]:
                    temp = current[1]
                    current[1] = current[3]
                    current[3] = temp

                draw.ellipse(current,outline=0) # not setting fill makes it transparent
            else:
                draw.line(individual[index:index+number_of_parameters],fill=0, width=2) # width=12 is supersampling
            index += number_of_parameters
            swap = not swap

    elif draw_settings == "selfchoice":
        while index < number_genes-(number_of_parameters-1):
            current = individual[index:index+number_of_parameters]
            tool = current[-1]
            if tool == -1:
                if current[0] > current[2]:
                    temp = current[0]
                    current[0] = current[2]
                    current[2] = temp
                if current[1] > current[3]:
                    temp = current[1]
                    current[1] = current[3]
                    current[3] = temp

                draw.ellipse(current[0:-1],outline=0) # not setting fill makes it transparent
            else:
                draw.line(individual[index:index+number_of_parameters-1],fill=0, width=2) # width=12 is supersampling
            index += number_of_parameters

    else:
        while index < number_genes-(number_of_parameters-1):
            draw.line(individual[index:index+number_of_parameters],fill=0, width=3) # width=12 is supersampling
            index += number_of_parameters
    iteration = str(iteration)
    while len(iteration) < 4:
        iteration = "0"+iteration
    im.save(os.path.join(saveddrawings,desired_text,iteration+".png"))

def fitness_evaluation(desired, classifier, net, transformer, base_image=None,individual=[0,0,30,10,40,40,100,100]):
    #im = Image.new("L", (256,256),(256))
    #im = Image.new("L", (1024,1024), (256))
    im = Image.fromarray(base_image)
    #individual = list(numpy.array(individual)*4) # lengthen lines for supersampling
    draw = ImageDraw.Draw(im)
    index = 0

    if draw_settings == "straight":
        while index < number_genes-(number_of_parameters-1):
            draw.line(individual[index:index+number_of_parameters],fill=0, width=3) # width=12 is supersampling
            index += number_of_parameters
    
    elif draw_settings == "ellipse":
        while index < number_genes-(number_of_parameters-1):
            current = individual[index:index+number_of_parameters]
            if current[0] > current[2]:
                temp = current[0]
                current[0] = current[2]
                current[2] = temp
            if current[1] > current[3]:
                temp = current[1]
                current[1] = current[3]
                current[3] = temp

            draw.ellipse(current,outline=0) # not setting fill makes it transparent
            index += number_of_parameters
    
    elif draw_settings == "strellipse":
        swap = False
        while index < number_genes-(number_of_parameters-1):
            current = individual[index:index+number_of_parameters]
            if swap:
                if current[0] > current[2]:
                    temp = current[0]
                    current[0] = current[2]
                    current[2] = temp
                if current[1] > current[3]:
                    temp = current[1]
                    current[1] = current[3]
                    current[3] = temp

                draw.ellipse(current,outline=0) # not setting fill makes it transparent
            else:
                draw.line(individual[index:index+number_of_parameters],fill=0, width=2) # width=12 is supersampling
            index += number_of_parameters
            swap = not swap

    elif draw_settings == "selfchoice":
        while index < number_genes-(number_of_parameters-1):
            current = individual[index:index+number_of_parameters]
            tool = current[-1]
            if tool == -1:
                if current[0] > current[2]:
                    temp = current[0]
                    current[0] = current[2]
                    current[2] = temp
                if current[1] > current[3]:
                    temp = current[1]
                    current[1] = current[3]
                    current[3] = temp

                draw.ellipse(current[0:-1],outline=0) # not setting fill makes it transparent
            else:
                draw.line(individual[index:index+number_of_parameters-1],fill=0, width=2) # width=12 is supersampling
            index += number_of_parameters

    else:
        while index < number_genes-(number_of_parameters-1):
            draw.line(individual[index:index+number_of_parameters],fill=0, width=3) # width=12 is supersampling
            index += number_of_parameters
    #im = im.resize((256,256),resample=Image.LANCZOS)
    #im.show()
    im2 = numpy.array(im)/255.0
    im2 = im2[:,:,numpy.newaxis]
    im2 = numpy.tile(im2,(1,1,3))
    
    net.blobs["data"].reshape(1,3,227,227)
    net.blobs["data"].data[...] = transformer.preprocess("data", im2)
    net.forward(end="fc6")
    #master_list5.append(net.blobs["pool5"].data.flatten())
    vector = net.blobs["fc6"].data.flatten()
    #master_list7.append(net.blobs["fc7"].data.flatten())

    #probs = classifier.predict_proba([vector])
    confidences = list(classifier.decision_function([vector])[0])
    raw = confidences[desired]
    
    #prediction = classifier.predict([vector])[0]
    #print prediction

    ranked_conf = [(value,i) for i,value in enumerate(confidences)]
    ranked_conf.sort()
    rank = ranked_conf.index((raw,desired))

    #print rank
    #raw = confidences[desired]
    #confidences.pop(desired)
    #relative = raw/sum(confidences)
    #return raw
    if raw < 0:
        return raw
    else:    
        return raw*rank

    #print probs[0]
    #print probs[0][desired]
    #print max(probs[0])
    #print list(probs[0]).index(max(probs[0]))

    #print len(confidences[0])
    #print confidences[0][desired]
    #print max(confidences[0]), min(confidences[0])
    #print list(confidences[0]).index(max(confidences[0]))
    #print classifier.predict([vector])
    

    """
    baseline = Image.new("L", (256,256), (256))
    draw = ImageDraw.Draw(baseline)
    draw.line([10,10,10,200], fill=0, width=3)
    draw.line([10,200,200,200], fill=0, width=3)
    draw.line([200,200,200,10], fill=0, width=3)
    draw.line([200,10,10,10], fill=0, width=3)
    baseline= numpy.array(baseline)/255.0
    
    score = numpy.sum((im2-baseline)**2)

    if score < 2000:
        im.show()
        crash
    """

def fitness_test(desired, classifier, net, transformer, population):
    base_image = Image.new("L", (256,256),(256)) # Supersampling at 1024 for antialiasing
    base_image = numpy.array(base_image)

    scores = [None]*population_size
    for i in range(population_size):
        scores[i] = (fitness_evaluation(desired, classifier, net, transformer, base_image=base_image,individual=population[i]),i)
    return scores

def generate_random():
    if draw_settings == "straight" or draw_settings == "ellipse" or draw_settings == "strellipse":
        individual = [random.randint(0,255) for _ in range(number_genes)]
    elif draw_settings == "selfchoice":
        individual = []
        for i in range(number_genes):
            if (i+1) % 5 == 0:
                individual.append(random.choice((-1,-2)))
            else:
                individual.append(random.randint(0,255))
    else:
        individual = [random.randint(0,255) for _ in range(number_genes)]

    return individual

def generate_population():
    population = [generate_random() for _ in range(population_size)]
    return population

def two_crossover(parent1, parent2):
    #max = number_genes-1
    maximum = number_of_strokes-1
    first = random.randint(1,maximum)
    second = random.randint(1,maximum)

    first = first*number_of_parameters
    second = second*number_of_parameters
    
    if first > second:
        temp = first
        first = second
        second = temp
        
    child1 = list(parent1)
    child2 = list(parent2)
    
    for index in range(first,second):
        if random.random() < mutation_probability:
            child1[index] = mutate(parent2[index])
        else:
            child1[index] = parent2[index]
        if random.random() < mutation_probability:
            child2[index] = mutate(parent1[index])
        else:
            child2[index] = parent1[index]
    
    return child1, child2

def one_crossover(parent1, parent2):
    #first = random.randint(1,number_genes-1)
    first = random.randint(1,number_of_strokes-1)
    first = first*number_of_parameters

    child1 = list(parent1)
    child2 = list(parent2)

    for index in range(first):
        if random.random() < mutation_probability:
            child1[index] = mutate(parent2[index])
        else:
            child1[index] = parent2[index]
        if random.random() < mutation_probability:
            child2[index] = mutate(parent1[index])
        else:
            child2[index] = parent1[index]
    
    return child1,child2

def mutate(allele):
    if random.random() < large_mutation_probability:
        if draw_settings == "straight" or draw_settings == "ellipse" or draw_settings == "strellipse":
            return random.randint(0,255)
        
        elif draw_settings == "selfchoice":
            if allele < 0:
                if allele == -1:
                    return -2
                else:
                    return -1
            else:
                return random.randint(0,255)
        
        else:
            return random.randint(0,255)
    else:
        if draw_settings == "straight" or draw_settings == "ellipse" or draw_settings == "strellipse":
            allele = allele+random.choice([-10,10])
            if allele > 254:
                allele = 254
            elif allele < 1:
                allele = 1
        
        elif draw_settings == "selfchoice":
            if allele < 0:
                return allele
            else:
                allele = allele+random.choice([-10,10])
                if allele > 254:
                    allele = 254
                elif allele < 1:
                    allele = 1
        else:
            allele = allele+random.choice([-10,10])
            if allele > 254:
                allele = 254
            elif allele < 1:
                allele = 1
        
        return allele

def all_mutate(individual):
    for i in range(number_genes):
        if random.random() < mutation_probability:
            individual[i] = mutate(individual[i])

    return individual

def generate_offsprings(selected, scores, population):
    new_population = [None]*population_size
    fittest = min_or_max(scores)
    best = list(population[fittest[1]])
    for i in range(population_size):
        if random.random() < crossover_probability and i < population_size-1:
            #new_population[i], new_population[i+1] = one_crossover(selected[i],selected[i+1])
            new_population[i], new_population[i+1] = two_crossover(selected[i],selected[i+1])
        else:
            new_population[i] = all_mutate(selected[i])
    new_population[0] = best # elitism
    return new_population

def genetic_loop():
    caffe.set_mode_gpu()

    classifier = joblib.load(os.path.abspath("/home/zebreu/Things/sketchfeatures/classifier1.pk1"))

    net = caffe.Net(model_file, pretrained, caffe.TEST)

    transformer = build_transformer(net)    

    labels = numpy.load(os.path.join(featureroot,"labels.npy"))
    labelslist = sorted(list(set(labels)))


    """
    desired = None
    print "\nHello! Enter 0 as your answer to get a list of objects I know about."
    desired_text = raw_input("What do you want me to draw?\n")
    while desired == None:
        if desired_text == "0":
            print labelslist
        try:
            desired = labelslist.index(desired_text)
        except:
            desired_text = raw_input("Sorry, I don't know what that is. Ask again?\n")
    print desired
    """
    
    for currentlabel in labelslist:
        desired_text = currentlabel
        desired = labelslist.index(desired_text)

        if not os.path.exists(os.path.join(saveddrawings,desired_text)):
            os.makedirs(os.path.join(saveddrawings,desired_text))

        population = generate_population()
        scores = fitness_test(desired, classifier, net, transformer, population)
        selected = selection(population, scores)
        for iteration in range(iterations):
            population = generate_offsprings(selected,scores,population)
            scores = fitness_test(desired, classifier, net, transformer, population)
            selected = selection(population, scores)
            fittest = min_or_max(scores)
            print "Best score: "+str(fittest[0])
            #print "Individual:", population[fittest[1]]
            if iteration % 10 == 0:
                draw_individual(population[fittest[1]],iteration,desired_text)
            print "Generation: "+str(iteration)
            #if fittest[0] < good_enough:
            #    break
        fittest = min_or_max(scores)
        print "Best score: "+str(fittest[0])
        print "Fittest individual: ", population[fittest[1]]
        draw_individual(population[fittest[1]],iteration,desired_text)

if __name__ == "__main__":
    genetic_loop()
