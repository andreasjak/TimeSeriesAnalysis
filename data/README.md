# Data

All data sets can be loaded in Python; MATLAB is not needed.

| File | Load with |
| --- | --- |
| `*.csv` | `pd.read_csv('../data/<file>.csv')` |
| `*.mat` | `scipy.io.loadmat('../data/<file>.mat')['<name>'].flatten()` |
| `*.dat` | `np.loadtxt('../data/<file>.dat')` |
| `*.wav` | `soundfile.read('../data/<file>.wav')` |

The `data*.m` files are the original MATLAB versions of the textbook data sets. Each has a CSV copy:

| MATLAB file | CSV file | Columns |
| --- | --- | --- |
| `dataBoxJenkinsA.m` | `boxjenkins_a.csv` | `concentration` |
| `dataCaffeine.m` | `caffeine.csv` | `caffeine` |
| `dataDowJones.m` | `dow_jones.csv` | `dow_jones`, `all_ordinaries` |
| `dataLydiaPinkham.m` | `lydia_pinkham.csv` | `Year`, `Advertising`, `Sales` |
| `dataMinkMuskrat.m` | `mink_muskrat.csv` | `year`, `mink`, `muskrat` |
| `dataMortality.m` | `mortality.csv` | `week_start`, `deaths` |
| `dataNewYorkBlackout.m` | `new_york_blackout.csv` | `births` |
| `dataOzone.m` | `ozone.csv` | `time`, `ozone`, `intv1`, `intv2`, `intv3` |
| `dataSunspots.m` | `sunspots.csv` | `year`, `month`, `sunspots` |
| `dataSweat.m` | `sweat.csv` | `sweat_rate`, `sodium`, `potassium` |
| `dataTobacco.m` | `tobacco_data.csv` | `year`, `production` |
| `dataWine.m` | `wine.csv` | `month`, `sales` |

Notes:
- `dataOzone.m` sets `data = data0(1:60, 2)`, so it uses only the first 60 months. `ozone.csv` holds all 216 rows.
- `dataDowJones.m` sets `data = data0(:, 1)`, which is the `dow_jones` column.
