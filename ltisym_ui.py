# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Sep  8 2010)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx

###########################################################################
## Class MainFrame
###########################################################################

class MainFrame ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"LTI Sym", pos = wx.DefaultPosition, size = wx.Size( 523,463 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		self.menu = wx.MenuBar( 0 )
		self.menu_file = wx.Menu()
		self.menu.Append( self.menu_file, u"File" ) 
		
		self.SetMenuBar( self.menu )
		
		self.m_statusBar1 = self.CreateStatusBar( 1, wx.ST_SIZEGRIP, wx.ID_ANY )
		szr_main = wx.BoxSizer( wx.VERTICAL )
		
		self.spl_main = wx.SplitterWindow( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SP_3D )
		self.spl_main.Bind( wx.EVT_IDLE, self.spl_mainOnIdle )
		
		self.pnl_top = wx.Panel( self.spl_main, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		szr_top = wx.BoxSizer( wx.VERTICAL )
		
		self.m_splitter3 = wx.SplitterWindow( self.pnl_top, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SP_3D )
		self.m_splitter3.Bind( wx.EVT_IDLE, self.m_splitter3OnIdle )
		
		self.pnl_left = wx.Panel( self.m_splitter3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		szr_comps = wx.FlexGridSizer( 1, 1, 0, 0 )
		szr_comps.AddGrowableCol( 0 )
		szr_comps.AddGrowableRow( 0 )
		szr_comps.SetFlexibleDirection( wx.BOTH )
		szr_comps.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		
		szr_comps = wx.StaticBoxSizer( wx.StaticBox( self.pnl_left, wx.ID_ANY, u"Components" ), wx.VERTICAL )
		
		szr_comps.Add( szr_comps, 1, wx.EXPAND, 5 )
		
		self.pnl_left.SetSizer( szr_comps )
		self.pnl_left.Layout()
		szr_comps.Fit( self.pnl_left )
		self.pnl_right = wx.Panel( self.m_splitter3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		szr_right = wx.FlexGridSizer( 1, 1, 0, 0 )
		szr_right.AddGrowableCol( 0 )
		szr_right.AddGrowableRow( 0 )
		szr_right.SetFlexibleDirection( wx.BOTH )
		szr_right.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		
		szr_equ = wx.StaticBoxSizer( wx.StaticBox( self.pnl_right, wx.ID_ANY, u"Expression" ), wx.VERTICAL )
		
		szr_right.Add( szr_equ, 1, wx.EXPAND, 5 )
		
		self.pnl_right.SetSizer( szr_right )
		self.pnl_right.Layout()
		szr_right.Fit( self.pnl_right )
		self.m_splitter3.SplitVertically( self.pnl_left, self.pnl_right, 0 )
		szr_top.Add( self.m_splitter3, 1, wx.EXPAND, 5 )
		
		self.pnl_top.SetSizer( szr_top )
		self.pnl_top.Layout()
		szr_top.Fit( self.pnl_top )
		self.pnl_bottom = wx.Panel( self.spl_main, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		szr_btm = wx.BoxSizer( wx.VERTICAL )
		
		self.ntb_btm = wx.Notebook( self.pnl_bottom, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.tab_output = wx.Panel( self.ntb_btm, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.ntb_btm.AddPage( self.tab_output, u"Output", False )
		
		szr_btm.Add( self.ntb_btm, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.pnl_bottom.SetSizer( szr_btm )
		self.pnl_bottom.Layout()
		szr_btm.Fit( self.pnl_bottom )
		self.spl_main.SplitHorizontally( self.pnl_top, self.pnl_bottom, 0 )
		szr_main.Add( self.spl_main, 1, wx.EXPAND, 5 )
		
		self.SetSizer( szr_main )
		self.Layout()
		
		self.Centre( wx.BOTH )
	
	def __del__( self ):
		pass
	
	def spl_mainOnIdle( self, event ):
		self.spl_main.SetSashPosition( 0 )
		self.spl_main.Unbind( wx.EVT_IDLE )
	
	def m_splitter3OnIdle( self, event ):
		self.m_splitter3.SetSashPosition( 0 )
		self.m_splitter3.Unbind( wx.EVT_IDLE )
	

