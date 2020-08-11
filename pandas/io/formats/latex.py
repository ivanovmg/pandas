"""
Module for formatting output data in Latex.
"""
from abc import ABC, abstractmethod
from typing import IO, List, Optional, Tuple

import numpy as np

from pandas.core.dtypes.generic import ABCMultiIndex

from pandas.io.formats.format import DataFrameFormatter, TableFormatter


class LatexFormatterAbstract(ABC):
    @abstractmethod
    def write_result(self, buf: IO[str]) -> None:
        """
        Render a DataFrame to a LaTeX tabular, longtable, or table/tabular
        environment output.
        """

    @abstractmethod
    def _write_env_begin(self, buf, column_format):
        pass

    @abstractmethod
    def _write_env_body(self, buf, strcols):
        pass

    @abstractmethod
    def _write_env_end(self, buf):
        pass

    @abstractmethod
    def _write_header_separator(self, buf):
        pass


class LatexFormatter(TableFormatter, LatexFormatterAbstract):
    """
    Used to render a DataFrame to a LaTeX tabular/longtable environment output.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
    column_format : str, default None
        The columns format as specified in `LaTeX table format
        <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g 'rcl' for 3 columns

    See Also
    --------
    HTMLFormatter
    """

    def __init__(
        self,
        formatter: DataFrameFormatter,
        column_format: Optional[str] = None,
        multicolumn: bool = False,
        multicolumn_format: Optional[str] = None,
        multirow: bool = False,
        caption: Optional[str] = None,
        short_caption: Optional[str] = None,
        label: Optional[str] = None,
        position: Optional[str] = None,
    ):
        self.fmt = formatter
        self.frame = self.fmt.frame
        self.bold_rows = self.fmt.bold_rows
        self.column_format = column_format
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.caption = caption
        self.short_caption = short_caption
        self.label = label
        self.escape = self.fmt.escape
        self.position = position

    def write_result(self, buf: IO[str]) -> None:
        self._write_env_begin(buf, self._get_column_format())
        self._write_env_body(buf, self._get_strcols())
        self._write_env_end(buf)

    def _write_header(self, buf, header):
        if header:
            buf.write(f"{header}\n")

    def _write_env_body(self, buf, strcols):
        ilevels = self.frame.index.nlevels
        clevels = self.frame.columns.nlevels
        nlevels = clevels
        if self.fmt.has_index_names and self.fmt.show_index_names:
            nlevels += 1
        strrows = list(zip(*strcols))
        self.clinebuf: List[List[int]] = []

        buf.write("\\toprule\n")
        for i, row in enumerate(strrows):
            if i == nlevels and self.fmt.header:
                self._write_header_separator(buf, row)

            crow = self._format_row(row, ilevels)

            if i < clevels and self.fmt.header and self.multicolumn:
                # sum up columns to multicolumns
                crow = self._format_multicolumn(crow, ilevels)
            if i >= nlevels and self.fmt.index and self.multirow and ilevels > 1:
                # sum up rows to multirows
                crow = self._format_multirow(crow, ilevels, i, strrows)
            buf.write(" & ".join(crow))
            buf.write(" \\\\\n")
            if self.multirow and i < len(strrows) - 1:
                self._print_cline(buf, i, len(strcols))

    def _format_row(self, row, ilevels):
        if self.escape:
            # escape backslashes first
            crow = [
                (
                    x.replace("\\", "\\textbackslash ")
                    .replace("_", "\\_")
                    .replace("%", "\\%")
                    .replace("$", "\\$")
                    .replace("#", "\\#")
                    .replace("{", "\\{")
                    .replace("}", "\\}")
                    .replace("~", "\\textasciitilde ")
                    .replace("^", "\\textasciicircum ")
                    .replace("&", "\\&")
                    if (x and x != "{}")
                    else "{}"
                )
                for x in row
            ]
        else:
            crow = [x if x else "{}" for x in row]
        if self.bold_rows and self.fmt.index:
            # bold row labels
            crow = [
                f"\\textbf{{{x}}}" if j < ilevels and x.strip() not in ["", "{}"] else x
                for j, x in enumerate(crow)
            ]
        return crow

    def _get_strcols(self):
        # string representation of the columns
        if len(self.frame.columns) == 0 or len(self.frame.index) == 0:
            info_line = (
                f"Empty {type(self.frame).__name__}\n"
                f"Columns: {self.frame.columns}\n"
                f"Index: {self.frame.index}"
            )
            strcols = [[info_line]]
        else:
            strcols = self.fmt._to_str_columns()

        # reestablish the MultiIndex that has been joined by _to_str_column
        if self.fmt.index and isinstance(self.frame.index, ABCMultiIndex):
            out = self.frame.index.format(
                adjoin=False,
                sparsify=self.fmt.sparsify,
                names=self.fmt.has_index_names,
                na_rep=self.fmt.na_rep,
            )

            # index.format will sparsify repeated entries with empty strings
            # so pad these with some empty space
            def pad_empties(x):
                for pad in reversed(x):
                    if pad:
                        break
                return [x[0]] + [i if i else " " * len(pad) for i in x[1:]]

            out = (pad_empties(i) for i in out)

            # Add empty spaces for each column level
            clevels = self.frame.columns.nlevels
            out = [[" " * len(i[-1])] * clevels + i for i in out]

            # Add the column names to the last index column
            cnames = self.frame.columns.names
            if any(cnames):
                new_names = [i if i else "{}" for i in cnames]
                out[self.frame.index.nlevels - 1][:clevels] = new_names

            # Get rid of old multiindex column and add new ones
            strcols = out + strcols[1:]
        return strcols

    def _get_column_format(self):
        def get_col_type(dtype):
            if issubclass(dtype.type, np.number):
                return "r"
            else:
                return "l"

        if self.column_format is None:
            dtypes = self.frame.dtypes._values
            column_format = "".join(map(get_col_type, dtypes))
            if self.fmt.index:
                index_format = "l" * self.frame.index.nlevels
                column_format = index_format + column_format
        elif not isinstance(self.column_format, str):  # pragma: no cover
            raise AssertionError(
                f"column_format must be str or unicode, not {type(column_format)}"
            )
        else:
            column_format = self.column_format
        return column_format

    def _format_multicolumn(self, row: List[str], ilevels: int) -> List[str]:
        r"""
        Combine columns belonging to a group to a single multicolumn entry
        according to self.multicolumn_format

        e.g.:
        a &  &  & b & c &
        will become
        \multicolumn{3}{l}{a} & b & \multicolumn{2}{l}{c}
        """
        row2 = list(row[:ilevels])
        ncol = 1
        coltext = ""

        def append_col():
            # write multicolumn if needed
            if ncol > 1:
                row2.append(
                    f"\\multicolumn{{{ncol:d}}}{{{self.multicolumn_format}}}"
                    f"{{{coltext.strip()}}}"
                )
            # don't modify where not needed
            else:
                row2.append(coltext)

        for c in row[ilevels:]:
            # if next col has text, write the previous
            if c.strip():
                if coltext:
                    append_col()
                coltext = c
                ncol = 1
            # if not, add it to the previous multicolumn
            else:
                ncol += 1
        # write last column name
        if coltext:
            append_col()
        return row2

    def _format_multirow(
        self, row: List[str], ilevels: int, i: int, rows: List[Tuple[str, ...]]
    ) -> List[str]:
        r"""
        Check following rows, whether row should be a multirow

        e.g.:     becomes:
        a & 0 &   \multirow{2}{*}{a} & 0 &
          & 1 &     & 1 &
        b & 0 &   \cline{1-2}
                  b & 0 &
        """
        for j in range(ilevels):
            if row[j].strip():
                nrow = 1
                for r in rows[i + 1 :]:
                    if not r[j].strip():
                        nrow += 1
                    else:
                        break
                if nrow > 1:
                    # overwrite non-multirow entry
                    row[j] = f"\\multirow{{{nrow:d}}}{{*}}{{{row[j].strip()}}}"
                    # save when to end the current block with \cline
                    self.clinebuf.append([i + nrow - 1, j + 1])
        return row

    def _print_cline(self, buf: IO[str], i: int, icol: int) -> None:
        """
        Print clines after multirow-blocks are finished.
        """
        for cl in self.clinebuf:
            if cl[0] == i:
                buf.write(f"\\cline{{{cl[1]:d}-{icol:d}}}\n")
        # remove entries that have been written to buffer
        self.clinebuf = [x for x in self.clinebuf if x[0] != i]

    def _compose_caption_macro(self):
        if self.caption is None:
            return ""
        if self.short_caption:
            return f"\\caption[{self.short_caption}]{{{self.caption}}}"
        return f"\\caption{{{self.caption}}}"

    def _compose_label_macro(self):
        if self.label is None:
            return ""
        else:
            return f"\\label{{{self.label}}}"

    def _get_position_option_string(self):
        if self.position is None:
            return ""
        else:
            return f"[{self.position}]"


class RegularCaptionMixin:
    def _compose_caption_and_label_macro(self):
        parts = [
            self._compose_caption_macro(),
            self._compose_label_macro(),
        ]
        return "\n".join([x for x in parts if x])


class LongTableCaptionMixin:
    def _compose_caption_and_label_macro(self):
        parts = [
            self._compose_caption_macro(),
            self._compose_label_macro(),
        ]
        caption_and_label = "".join([x for x in parts if x])
        if caption_and_label:
            # a double-backslash is required at the end of the line
            # as discussed here:
            # https://tex.stackexchange.com/questions/219138
            double_backslash = "\\\\"
            return "".join([caption_and_label, double_backslash])


class LatexTableFormatter(LatexFormatter, RegularCaptionMixin):
    def _write_env_begin(self, buf, column_format: str):
        """
        Write the beginning of a table environment and
        nested tabular environments including caption and label.

        Parameters
        ----------
        buf : string or file handle
            File path or object. If not specified, the result is returned as
            a string.
        column_format : str
            The columns format as specified in `LaTeX table format
            <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g 'rcl'
            for 3 columns
        """
        position_ = self._get_position_option_string()
        buf.write(f"\\begin{{table}}{position_}\n\\centering\n")

        caption_and_label = self._compose_caption_and_label_macro()
        if caption_and_label:
            buf.write(f"{caption_and_label}\n")

        buf.write(f"\\begin{{tabular}}{{{column_format}}}\n")

    def _write_env_end(self, buf):
        """
        Write the end of a table environment.

        Parameters
        ----------
        buf : string or file handle
            File path or object. If not specified, the result is returned as
            a string.

        """
        buf.write("\\bottomrule\n")
        buf.write("\\end{tabular}\n")
        buf.write("\\end{table}\n")

    def _write_header_separator(self, buf, row):
        buf.write("\\midrule\n")


class LatexTabularFormatter(LatexFormatter, RegularCaptionMixin):
    def _write_env_begin(self, buf, column_format: str):
        """
        Write the beginning of a tabular environment.

        Parameters
        ----------
        buf : string or file handle
            File path or object. If not specified, the result is returned as
            a string.
        column_format : str
            The columns format as specified in `LaTeX table format
            <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g 'rcl'
            for 3 columns
        """
        buf.write(f"\\begin{{tabular}}{{{column_format}}}\n")

    def _write_env_end(self, buf):
        """
        Write the end of a tabular environment or nested table/tabular
        environment.

        Parameters
        ----------
        buf : string or file handle
            File path or object. If not specified, the result is returned as
            a string.

        """
        buf.write("\\bottomrule\n")
        buf.write("\\end{tabular}\n")

    def _write_header_separator(self, buf, row):
        buf.write("\\midrule\n")


class LatexLongTableFormatter(LatexFormatter, LongTableCaptionMixin):
    def _write_env_begin(self, buf, column_format: str):
        """
        Write the beginning of a longtable environment including caption and
        label if provided by user.

        Parameters
        ----------
        buf : string or file handle
            File path or object. If not specified, the result is returned as
            a string.
        column_format : str
            The columns format as specified in `LaTeX table format
            <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g 'rcl'
            for 3 columns
        """
        position_ = self._get_position_option_string()
        buf.write(f"\\begin{{longtable}}{position_}{{{column_format}}}\n")

        caption_and_label = self._compose_caption_and_label_macro()
        if caption_and_label:
            buf.write(f"{caption_and_label}\n")

    def _write_env_end(self, buf):
        """
        Write the end of a longtable environment.

        Parameters
        ----------
        buf : string or file handle
            File path or object. If not specified, the result is returned as
            a string.

        """
        buf.write("\\end{longtable}\n")

    def _write_header_separator(self, buf, row):
        buf.write("\\midrule\n")
        buf.write("\\endhead\n")
        buf.write("\\midrule\n")
        buf.write(
            f"\\multicolumn{{{len(row)}}}{{r}}" "{{Continued on next page}} \\\\\n"
        )
        buf.write("\\midrule\n")
        buf.write("\\endfoot\n\n")
        buf.write("\\bottomrule\n")
        buf.write("\\endlastfoot\n")
