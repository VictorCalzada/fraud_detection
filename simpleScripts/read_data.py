import pandas as pd


FILENAME = "data/socec_data.csv"


def get_data(iso_prov, dt, socec_df=pd.read_csv(FILENAME)):
    """
    Get the closest to date socioeconomic data value available

    :param iso_prov: iso code of spanish subregion
    :param dt: date

    :return: pandas.Serie
    """

    yr = dt.year
    socec_df = socec_df[socec_df["iso_prov"] == iso_prov]

    to_return = pd.DataFrame({"year": [yr]})
    request_col = [col for col in socec_df.columns if col not in
                   to_return.columns]

    while request_col and (yr >= socec_df.year.min()):
        to_return = pd.concat([to_return.reset_index(drop=True),
                               socec_df.loc[socec_df["year"] == yr, :] \
                                       .loc[:, request_col] \
                                       .reset_index(drop=True)
                               ], axis=1) \
                      .dropna(axis=1, how="all")
        yr -= 1
        request_col = [col for col in socec_df.columns if col not in
                       to_return.columns]
    else:
        if request_col:
            print(iso_prov, request_col, dt.year, sep=" / ")

    return pd.concat([pd.DataFrame(columns=socec_df.columns),
                      to_return], axis=0)


if __name__ == '__main__':
    from datetime import date

    print(get_data(39, date(2019, 2, 28)))
    print(get_data(8, date(2022, 1, 1)))
