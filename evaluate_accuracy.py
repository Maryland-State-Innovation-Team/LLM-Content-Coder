import argparse
import pandas as pd
import click
import numpy as np


def main():
    parser = argparse.ArgumentParser(
                    prog='Accuracy evaluation',
                    description='Evaluate LLM classify accuracy')
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-n', '--count', type=int, default=100)
    parser.add_argument('-d', '--description', type=int, default=1)
    parser.add_argument('-r', '--reasonings', type=int, default=2)
    parser.add_argument('-c', '--classifications', type=int, default=3)
    args = parser.parse_args()

    data = pd.read_csv(args.filename)
    data = data.sample(n=args.count).reset_index(drop=True)

    colnames = data.columns
    descriptionColumnName = colnames[args.description - 1]
    reasoningsColumnName = colnames[args.reasonings - 1]
    classificationsColumnName = colnames[args.classifications - 1]
    descriptionColumn = data[[descriptionColumnName]]
    reasoningsColumn = data[[reasoningsColumnName]]
    classificationsColumn = data[[classificationsColumnName]]
    descriptionColumnValues = [val[0] for val in descriptionColumn.values.tolist()]
    reasoningsColumnValues = [val[0] for val in reasoningsColumn.values.tolist()]
    classificationsColumnValues = [val[0] for val in classificationsColumn.values.tolist()]
    classificationsCorrects = []
    for index, descriptionColumnValue in enumerate(descriptionColumnValues):
        reasoningsColumnValue = reasoningsColumnValues[index]
        classificationsColumnValue = classificationsColumnValues[index]
        print(f"\n\n{index + 1}. {descriptionColumnValue}\n\nReasoning: {reasoningsColumnValue}\n")
        try:
            classificationsCorrect = int(click.confirm(classificationsColumnValue, default=True))
            classificationsCorrects.append(classificationsCorrect)
        except click.exceptions.Abort:
            break
    sample_accuracy = np.mean(classificationsCorrects).tolist()
    sample_accuracy_round = round(sample_accuracy * 100, 1)
    print(f"Accuracy: {sample_accuracy_round}%")
        
        

if __name__ == '__main__':
    main()