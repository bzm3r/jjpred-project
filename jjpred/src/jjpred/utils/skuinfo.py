import sys
from typing import Literal
import polars as pl


from jjpred.analysisdefn import AnalysisDefn
from jjpred.database import DataBase
from jjpred.sku import Sku
from jjpred.structlike import MemberType
from jjpred.utils.fileio import read_meta_info
from jjpred.utils.polars import find_dupes
from jjpred.utils.typ import as_polars_type
