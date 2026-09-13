# Datasets

All data used across this repository's 14 practicals is real. Nothing is invented.

| File | What it is | Real source |
|---|---|---|
| `titanic.csv` | 891 real Titanic passengers, 1912 | Public GitHub mirror of the real historical passenger manifest |
| `student-mat.csv` | 395 real Portuguese secondary students | UCI Machine Learning Repository, Cortez & Silva (2008), Student Performance dataset |

## Titanic dataset

Real columns include `Survived` (0/1), `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, and more. Genuinely contains real missing values in `Age` (177), `Cabin` (687), and `Embarked` (2) — used throughout the course to teach real preprocessing.

## Student Performance dataset

Real columns include demographic, social, and school-related features, plus 3 real grade checkpoints: `G1`, `G2` (first and second period grades) and `G3` (final grade), each on a real 0-20 scale. Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance), licensed CC BY 4.0.

## Re-downloading the data

```bash
curl -o titanic.csv "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
curl -o student-mat.csv "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-mat.csv"
```

Note: the student performance CSV as hosted uses semicolon separators and quoted string fields — this repository's copy has already been cleaned to standard comma-separated, unquoted format for direct use with `pd.read_csv()`.
