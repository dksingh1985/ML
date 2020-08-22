from model import *

BATCH_SIZE = rows-30

model = get_model()

#model = load_model("2Sigma.dhf5")

net_in, net_out = get_training_data(BATCH_SIZE)
       
for x in range(100000000):
    for i in range(99999999999999999999999999999):
        print("i/x ---> ", i,"/", x)
        model.fit(net_in, net_out, batch_size=10, epochs=1, shuffle=False)
        if (i % 1000 == 0):
            model.save("2Sigma.dhf5", overwrite=True)
    



