import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

file_path = "./data/"
file_name = "3009680_p4.csv"

df = pd.read_csv(f"{file_path}{file_name}")

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

for col in df.columns:
    if col == "P":
        fig_title = "Precipitation over 2009-2013"
        x_label = "Date"
        y_label = "Precipitation [mm/day]"
    elif col == "Q":
        fig_title = "Discharge over 2009-2013"
        x_label = "Date"
        y_label = "Discharge [m³]"
    elif col == "AET":
        fig_title = "Actual Evaporatranspiration over 2009-2013"
        x_label = "Date"
        y_label = "AET [mm/day]"
    else:
        raise KeyError

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#101010", dpi=200)
    ax.patch.set_facecolor('#161616')

    ax.plot(df.index, df[col], color='#EAEAFA')

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 6, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    ax.grid(which='major', alpha=0.4)
    ax.grid(which='minor', alpha=0.2)

    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')

    ax.set_title(fig_title, color='white')
    ax.set_xlabel(x_label, color='white')
    ax.set_ylabel(y_label, color='white')
    plt.tight_layout()
    plt.savefig(f"./fig/{fig_title.replace(" ", "_")}.png")