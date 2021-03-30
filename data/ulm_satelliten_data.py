import pandas as pd

path = __file__.replace(".py", ".xlsx")
df = pd.read_excel(path)