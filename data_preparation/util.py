def flatten(data, parent_key='', sep='.'):
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten(v, new_key, sep=sep).items())
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, data))
    return dict(items)

def snake_to_camel(s: str) -> str:
    parts = s.split('_')
    return parts[0] + "".join([word.title() for word in parts])

"""
 FEATURE NAMES:
 https://www.uniprot.org/help/return_fields
"""

FIELDS = {
    "cc_function": {
        "primary_accession": "0.primaryAccession",
        "function": "0.comments.0.texts.0.value",
        "uniParcId": "0.extraAttributes.uniParcId"
    },

    "ec": {
        "ec_number": "0.proteinDescription.recommendedName.ecNumbers.0.value",
        "recommended_name": "0.proteinDescription.recommendedName.fullName.value",
        "alternative_name": "0.proteinDescription.alternativeNames.0.fullName.value",
        "flag": "0.proteinDescription.flag"
    },

    "cc_cofactor": {
        "cofactor": "0.comments.1.cofactors.0.name",
        "cofactor_chebi_id": "0.comments.1.cofactors.0.cofactorCrossReference.id",
        "cofactor_note": "0.comments.1.note.texts.0.value"
    },

    "protein_families": {
        "family_name": "0.comments.2.texts.0.value",
        "comment_type": "0.comments.2.commentType"
    },

    "go_f": {
        "go_ids": [
            "0.uniProtKBCrossReferences.2.id",
            "0.uniProtKBCrossReferences.3.id",
            "0.uniProtKBCrossReferences.4.id"
        ],
        "go_terms": [
            "0.uniProtKBCrossReferences.2.properties.0.value",
            "0.uniProtKBCrossReferences.3.properties.0.value",
            "0.uniProtKBCrossReferences.4.properties.0.value"
        ],
        "go_evidence": [
            "0.uniProtKBCrossReferences.2.properties.1.value",
            "0.uniProtKBCrossReferences.3.properties.1.value",
            "0.uniProtKBCrossReferences.4.properties.1.value"
        ]
    },

    "go_p": {
        "go_ids": [
            "0.uniProtKBCrossReferences.5.id",
            "0.uniProtKBCrossReferences.6.id",
            "0.uniProtKBCrossReferences.7.id"
        ],
        "go_terms": [
            "0.uniProtKBCrossReferences.5.properties.0.value",
            "0.uniProtKBCrossReferences.6.properties.0.value",
            "0.uniProtKBCrossReferences.7.properties.0.value"
        ],
        "go_evidence": [
            "0.uniProtKBCrossReferences.5.properties.1.value",
            "0.uniProtKBCrossReferences.6.properties.1.value",
            "0.uniProtKBCrossReferences.7.properties.1.value"
        ]
    }
}


field_names = list(FIELDS.keys())
