import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_pickle("../../data/scivqa_data_3000_updated_2024-09-01.pkl")
    papers_dist = df.groupby(["year", "venue"]).size().reset_index(name="count")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=papers_dist, x="year", y="count", hue="venue")
    plt.ylabel("Number of papers")
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.legend(title="Venue", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("papers_dist_3000.png")


if __name__ == "__main__":
    main()
