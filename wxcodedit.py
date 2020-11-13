"""07/05/17  Python code editor control with syntax highlighting and folding"""

from __future__ import print_function, division

__pyspec = 1

import keyword
import wx
import wx.stc


class PssPdCodeEditor(wx.stc.StyledTextCtrl):

    def __init__(self, parent, ID=wx.ID_ANY):

        super(PssPdCodeEditor, self).__init__(parent, ID,
                                              wx.DefaultPosition,
                                              wx.DefaultSize,
                                              wx.BORDER_NONE)

        self.CmdKeyAssign(ord('B'), wx.stc.STC_SCMOD_CTRL,
                          wx.stc.STC_CMD_ZOOMIN)

        self.CmdKeyAssign(ord('N'), wx.stc.STC_SCMOD_CTRL,
                          wx.stc.STC_CMD_ZOOMOUT)

        self.SetEdgeMode(wx.stc.STC_EDGE_BACKGROUND)

        self.SetEdgeColumn(79)

        self.Bind(wx.stc.EVT_STC_MARGINCLICK, self.on_margin_click)

        self.record = None

        self.setup()

    def setup(self):

        """This method carries out the work of setting up the editor.
        It's seperate so as not to clutter up the init code.
        """

        # set the lexer (python is built-in):
        self.SetLexer(wx.stc.STC_LEX_PYTHON)

        # set the keywords:
        self.SetKeyWords(0, " ".join(keyword.kwlist))

        # enable folding:
        self.SetProperty("fold", "1")

        # highlight tab/space mixing (shouldn't be any):
        self.SetProperty("tab.timmy.whinge.level", "1")

        # set left and right margins:
        self.SetMargins(2, 2)

        # set up the numbers in the margin for margin #1:
        self.SetMarginType(1, wx.stc.STC_MARGIN_NUMBER)

        # reasonable value for, say, 2-3 digits using a mono font:
        self.SetMarginWidth(1, 30)

        # indentation and tab stuff:
        self.SetIndent(4)  # prescribed indent size for python
        self.SetIndentationGuides(True)  # show indent guides
        self.SetBackSpaceUnIndents(True)

        # backspace unindents rather than delete 1 space:
        self.SetTabIndents(True)  # Tab key indents
        self.SetTabWidth(4)  # Proscribed tab size for wx
        self.SetUseTabs(False)  # Use spaces rather than tabs

        # don't show white space:
        self.SetViewWhiteSpace(False)

        # EOL: Since we are loading/saving ourselves, and the
        # strings will always have \n's in them, set the STC to
        # edit them that way.
        self.SetEOLMode(wx.stc.STC_EOL_LF)
        self.SetViewEOL(False)

        # no right-edge mode indicator:
        self.SetEdgeMode(wx.stc.STC_EDGE_NONE)

        # set up a margin to hold fold markers:
        self.SetMarginType(2, wx.stc.STC_MARGIN_SYMBOL)
        self.SetMarginMask(2, wx.stc.STC_MASK_FOLDERS)
        self.SetMarginSensitive(2, True)
        self.SetMarginWidth(2, 12)

        # and now set up the fold markers
        self.MarkerDefine(wx.stc.STC_MARKNUM_FOLDEREND,
                          wx.stc.STC_MARK_BOXPLUSCONNECTED, "white", "black")

        self.MarkerDefine(wx.stc.STC_MARKNUM_FOLDEROPENMID,
                          wx.stc.STC_MARK_BOXMINUSCONNECTED, "white", "black")

        self.MarkerDefine(wx.stc.STC_MARKNUM_FOLDERMIDTAIL,
                          wx.stc.STC_MARK_TCORNER, "white", "black")

        self.MarkerDefine(wx.stc.STC_MARKNUM_FOLDERTAIL,
                          wx.stc.STC_MARK_LCORNER, "white", "black")

        self.MarkerDefine(wx.stc.STC_MARKNUM_FOLDERSUB,
                          wx.stc.STC_MARK_VLINE, "white", "black")

        self.MarkerDefine(wx.stc.STC_MARKNUM_FOLDER,
                          wx.stc.STC_MARK_BOXPLUS, "white", "black")

        self.MarkerDefine(wx.stc.STC_MARKNUM_FOLDEROPEN,
                          wx.stc.STC_MARK_BOXMINUS, "white", "black")

        # global default style:
        defstyle = "fore:#000000,back:#FFFFFF,face:Consolas,size:12"
        self.StyleSetSpec(wx.stc.STC_STYLE_DEFAULT, defstyle)

        # clear styles and revert to default.
        self.StyleClearAll()

        # customize the style spec:
        self.StyleSetSpec(wx.stc.STC_STYLE_LINENUMBER,
                          'fore:#000000,back:#DDDDDD')

        self.StyleSetSpec(wx.stc.STC_STYLE_BRACELIGHT,
                          'fore:#00009D,back:#FFFF00')

        self.StyleSetSpec(wx.stc.STC_STYLE_BRACEBAD,
                          'fore:#00009D,back:#FF0000')

        self.StyleSetSpec(wx.stc.STC_STYLE_INDENTGUIDE, "fore:#CDCDCD")

        self.StyleSetSpec(wx.stc.STC_P_DEFAULT, 'fore:#000000')

        self.StyleSetSpec(wx.stc.STC_P_COMMENTLINE,
                          'fore:#008000,back:#FFFFFF')

        self.StyleSetSpec(wx.stc.STC_P_COMMENTBLOCK,
                          'fore:#008000,back:#FFFFFF')

        self.StyleSetSpec(wx.stc.STC_P_NUMBER, 'fore:#008080')

        self.StyleSetSpec(wx.stc.STC_P_STRING, 'fore:#800080')

        self.StyleSetSpec(wx.stc.STC_P_CHARACTER, 'fore:#800080')

        self.StyleSetSpec(wx.stc.STC_P_WORD, 'fore:#000080,bold')

        self.StyleSetSpec(wx.stc.STC_P_TRIPLE, 'fore:#800080,back:#FFFFFF')

        self.StyleSetSpec(wx.stc.STC_P_TRIPLEDOUBLE,
                          'fore:#800080,back:#FFFFFF')

        self.StyleSetSpec(wx.stc.STC_P_CLASSNAME, 'fore:#0000FF,bold')

        self.StyleSetSpec(wx.stc.STC_P_DEFNAME, 'fore:#008080,bold')

        self.StyleSetSpec(wx.stc.STC_P_OPERATOR, 'fore:#800000,bold')

        self.StyleSetSpec(wx.stc.STC_P_IDENTIFIER, 'fore:#000000')

        self.SetCaretForeground("BLUE")

        self.SetSelBackground(True,
                              wx.SystemSettings.GetColour(
                              wx.SYS_COLOUR_HIGHLIGHT))

        self.SetSelForeground(True,
                              wx.SystemSettings.GetColour(
                              wx.SYS_COLOUR_HIGHLIGHTTEXT))
        self.SetReadOnly(0)

    def on_margin_click(self, event):

        # fold and unfold as needed
        if event.GetMargin() == 2:

            if event.GetShift() and event.GetControl():
                self.fold_all()

            else:
                line_clicked = self.LineFromPosition(event.GetPosition())

                if (self.GetFoldLevel(line_clicked) &
                        wx.stc.STC_FOLDLEVELHEADERFLAG):

                    if event.GetShift():
                        self.SetFoldExpanded(line_clicked, True)
                        self.expand(line_clicked, True, True, 1)

                    elif event.GetControl():

                        if self.GetFoldExpanded(line_clicked):
                            self.SetFoldExpanded(line_clicked, False)
                            self.expand(line_clicked, False, True, 0)

                        else:
                            self.SetFoldExpanded(line_clicked, True)
                            self.expand(line_clicked, True, True, 100)
                    else:
                        self.ToggleFold(line_clicked)

    def fold_all(self):

        line_count = self.GetLineCount()
        expanding = True

        # find out if we are folding or unfolding:
        for linenum in range(line_count):

            if self.GetFoldLevel(linenum) & wx.stc.STC_FOLDLEVELHEADERFLAG:

                expanding = not self.GetFoldExpanded(linenum)
                break

        linenum = 0
        while linenum < line_count:

            level = self.GetFoldLevel(linenum)

            if (level & wx.stc.STC_FOLDLEVELHEADERFLAG and
                   (level & wx.stc.STC_FOLDLEVELNUMBERMASK) ==
                    wx.stc.STC_FOLDLEVELBASE):

                if expanding:
                    self.SetFoldExpanded(linenum, True)
                    linenum = self.expand(linenum, True)
                    linenum -= 1

                else:
                    last_child = self.GetLastChild(linenum, -1)
                    self.SetFoldExpanded(linenum, False)
                    if last_child > linenum:
                        self.HideLines(linenum + 1, last_child)

            linenum += 1

    def expand(self, line, doexpand, force=None, vislevels=0, level=-1):

        last_child = self.GetLastChild(line, level)
        line += 1

        if not force:
            force = False

        while line <= last_child:

            if force:
                if vislevels > 0:
                    self.ShowLines(line, line)
                else:
                    self.HideLines(line, line)
            else:
                if doexpand:
                    self.ShowLines(line, line)

            if level == -1:
                level = self.GetFoldLevel(line)

            if level & wx.stc.STC_FOLDLEVELHEADERFLAG:

                if force:
                    if vislevels > 1:
                        self.SetFoldExpanded(line, True)
                    else:
                        self.SetFoldExpanded(line, False)
                    line = self.expand(line, doexpand, force, vislevels - 1)

                else:
                    if doexpand and self.GetFoldExpanded(line):
                        line = self.expand(line, True, force, vislevels - 1)
                    else:
                        line = self.expand(line, False, force, vislevels - 1)

            else:
                line += 1

        return line