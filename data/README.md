# Data notes

The associated article used a public mineralogical dataset assembled from public sources including mineral databases and a Kaggle mineral dataset.

The raw dataset is not duplicated here. Obtain it from the original public source and verify its current license and schema.

## Expected columns

The statistical benchmark requires:

```text
Specific Gravity
Refractive Index
Molar Mass
```

The modelling matrix may additionally include numeric physical properties, encoded structural/optical properties and elemental-composition columns.

A mineral-name column such as `Name` should be retained for interpretation but excluded from the numeric model matrix.

## Licensing

The published article is CC BY-NC-ND 4.0. This repository contains an independent software implementation and does not redistribute the article PDF or article text.
