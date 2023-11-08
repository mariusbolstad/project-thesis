# project-thesis

Solstorm:

Login: 

```bash
ssh solstorm-login.iot.ntnu.no -l mariumbo
```

If you need to swap file:

Solstorm terminal:

```bash
rm main.py
```

Local terminal:
```bash
scp /Users/mariusbolstad/VSCode/project-thesis/main.py mariumbo@solstorm-login.iot.ntnu.no:~
scp mariumbo@solstorm-login.iot.ntnu.no:/storage/users/mariumbo/log2.csv /Users/mariusbolstad/VSCode/project-thesis/data/
```
```bash
scp -r /Users/mariusbolstad/VSCode/project-thesis/main.py /Users/mariusbolstad/VSCode/project-thesis/models/ /Users/mariusbolstad/VSCode/project-thesis/data/ /Users/mariusbolstad/VSCode/project-thesis/helper_functions.py  mariumbo@solstorm-login.iot.ntnu.no:~
```

Deploy program to an idle node (swap with 2-40): https://solstorm.iot.ntnu.no/ganglia/?c=Solstorm&m=load_one&r=hour&s=by%20name&hc=5&mc=2

Solstorm terminal:

```bash
rm -rf main.py models/ data/ helper_functions.py requirements.txt
screen
screen -ls
screen -rd <session number from screen -ls>
ssh compute-2-40
module load Python

```


Forecast results:

Daily BACI, 1-step ahead, 50 forecasts, ANN(8,1) 10 epochs, ARIMA(3,1,3)

ARIMA RMSEs: 77.28721338854554
Random Walk RMSEs: 105.8
ANN RMSEs: 157.4576513671875


Montly BACI, 1-step ahead, 50 forecasts, ANN(2,2) 10 epochs, ARIMA(1,1,1)

ARIMA RMSEs: 956.8166128664863
Random Walk RMSEs: 917.42
ANN RMSEs: 2885.1269592285157

ARIMA RMSEs: 964.6714350634938
Random Walk RMSEs: 925.5
ANN RMSEs: 946.1920088291168



Daily BACI, 1-step ahead, 50 forecasts, ANN(2,2) 10 epochs, ARIMA(1,1,2)

ARIMA RMSEs: 39.605423178044255
Random Walk RMSEs: 93.58
ANN RMSEs: 189.19609375

ARIMA RMSEs: 94.55155145466632
Random Walk RMSEs: 146.8
ANN RMSEs: 160.90745239257814


Daily BACI, 1-step ahead, 20 forecasts, ANN(1,2) 10 epochs, ARIMA(1,1,2)

ARIMA RMSEs: 94.55155145466632
Random Walk RMSEs: 146.8
ANN RMSEs: 138.44936828613282



Daily BACI, 1-step ahead, 50 forecasts. ANN(1,2) 3 epochs, ARIMA(1,1,2)
ARIMA RMSEs: 26.971779961488878
Random Walk RMSEs: 48.64
ANN RMSEs: 29.4990869140625



Best models:

ANN(2,1)
ANN(4,3)
ANN(4,2)
ANN(4,5)
ANN(1,6) (not exog diff)