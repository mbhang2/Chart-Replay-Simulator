import data_parser as dp
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui


df = dp.to_df("./NQ_tick_data/December_247.txt", to_timestamp=False)
dp.convert_datetime(df)
df.set_index("Time", inplace=True)

current_bar = 0
capital = 100000
position = 0.0

buy_price=[]
sell_price=[]

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.div(
                {"style": "font-weight: bold; color: black;"},
                ui.output_text("capital_title")
            ),
            ui.div(
                {"style": "font-weight: bold; color: black;"},
                ui.output_text_verbatim("compute_capital")
            ),
            ui.div(
                {"style": "font-weight: bold; color: black;"},
                ui.output_text("total_pnl_title")
            ),
            ui.div(
                {"style": "font-weight: bold; color: black;"},
                ui.output_text_verbatim("compute_total_pnl")
            ),
            ui.div(
                {"style": "font-weight: bold; color: black;"},
                ui.output_text("unrealized_pnl_title")
            ),
            ui.div(
                {"style": "font-weight: bold; color: black;"},
                ui.output_text_verbatim("compute_unrealized_pnl")
            ),
            ui.div(
                ui.help_text("-------------------------------------------------")
            ),
            ui.div(
                {"style": "font-weight: bold; color: gray;"},
                ui.output_text("curr_time")
            ),
            ui.div(
                {"style": "font-weight: bold; color: black;"},
                ui.output_text("curr_price")
            ),
            ui.div(
                ui.output_text("append_buy_price"),
                ui.output_text("append_sell_price"),
                ui.row(
                    {"style": "font-weight: bold; color: gray;"},
                    ui.help_text("보유 잔고")
                ),
                ui.output_text_verbatim("net_position"),
                ui.row(
                    {"style": "font-weight: bold; color: gray;"},
                    ui.help_text("평단가")
                ),
                ui.output_text_verbatim("avg_price")
            ),
            ui.row(
                ui.input_action_button("buy", "매수", width="100px", class_="btn-danger"),
                ui.input_action_button("sell", "매도", width="100px", class_="btn-primary"),
            ),
            width=2
        ),
        ui.panel_main(
            ui.output_plot("plot", '1200px', '720px')
        )
    ),
    ui.row(
        ui.input_text("skip", "첫 봉 번호", "3000"),
        ui.input_text("bars_to_show", "차트 봉 개수", "100"),
        ui.input_action_button("forward", "다음 봉"),
    )
)


def server(input, output, session):
    @output
    @render.text
    def capital_title():
        return "평가담보금"
    
    @render.text
    def total_pnl_title():
        return "총평가손익"
    
    @render.text
    def unrealized_pnl_title():
        return "미실현 손익"
    
    @render.text
    def curr_time():
        return f"{df.index[int(input.skip())+input.forward()+int(input.bars_to_show())]}"
    
    @render.text
    def curr_price():
        return f"현재가격: {df['Close'][int(input.skip())+input.forward()+int(input.bars_to_show())]}"
    
    @render.plot
    def plot():
        fig = mpf.figure(figsize=(12,9))
        ax = fig.add_subplot(1,1,1)

        tcdf = df['CL']
        apd  = mpf.make_addplot(tcdf, ax=ax)

        market_colors = mpf.make_marketcolors(up = 'red', down = 'blue')
        custom_style = mpf.make_mpf_style(marketcolors = market_colors, facecolor='white', figcolor='white', gridstyle='')

        mpf.plot(df[int(input.skip())+input.forward():int(input.skip())+input.forward()+int(input.bars_to_show())+1], volume=False, ax=ax, type='candle', style=custom_style, mav=(20))

        return fig
    
    @render.text
    @reactive.event(input.buy, ignore_none=False)
    def append_buy_price():
        buy_price.append(df["Close"][int(input.skip())+input.forward()+int(input.bars_to_show())])
        return f""
    
    @render.text
    @reactive.event(input.sell, ignore_none=False)
    def append_sell_price():
        sell_price.append(df["Close"][int(input.skip())+input.forward()+int(input.bars_to_show())])
        return f""
    
    @render.text
    @reactive.event(input.buy, input.sell, input.reset, ignore_none=False)
    def avg_price():
        net_pos = len(buy_price)-len(sell_price)

        if net_pos == 0:
            return f"0"

        return f"{round(np.average(buy_price[-abs(net_pos):]) if net_pos > 0 else np.average(sell_price[-abs(net_pos):]), 2)}"
    
    @render.text
    @reactive.event(input.buy, input.sell, input.reset, ignore_none=False)
    def net_position():
        net_pos = len(buy_price)-len(sell_price)

        return f"{net_pos}"
        
    @render.text
    @reactive.event(input.forward, input.reset, ignore_none=False)
    def compute_capital():
        net_pos = len(buy_price)-len(sell_price)
        min_len = min(len(buy_price), len(sell_price))
        realized_pnl = 0
        unrealized_pnl = 0
        
        for i in range(min_len):
            realized_pnl += (sell_price[i] - buy_price[i])*20

        if net_pos == 0:
            unrealized_pnl = 0
        
        curr_price = df["Close"][int(input.skip())+input.forward()+int(input.bars_to_show())]
        avg_price = np.average(buy_price[-abs(net_pos):]) if net_pos > 0 else np.average(sell_price[-abs(net_pos):])

        if net_pos > 0:
            unrealized_pnl = (curr_price-avg_price)*20*abs(net_pos)
        else:
            unrealized_pnl = (-curr_price+avg_price)*20*abs(net_pos)

        to_return = round(capital + realized_pnl + unrealized_pnl,2)

        if to_return >= 0:
            return f"${to_return:,}"
        else:
            return f"-${abs(to_return):,}"
        
    @render.text
    @reactive.event(input.buy, input.sell, input.forward, input.reset, ignore_none=False)
    def compute_total_pnl():
        min_len = min(len(buy_price), len(sell_price))
        realized_pnl = 0
        unrealized_pnl = 0
        
        for i in range(min_len):
            realized_pnl += (sell_price[i] - buy_price[i])*20

        net_pos = len(buy_price)-len(sell_price)

        if net_pos == 0:
            unrealized_pnl = 0
        
        curr_price = df["Close"][int(input.skip())+input.forward()+int(input.bars_to_show())]
        avg_price = np.average(buy_price[-abs(net_pos):]) if net_pos > 0 else np.average(sell_price[-abs(net_pos):])

        if net_pos > 0:
            unrealized_pnl = (curr_price-avg_price)*20*abs(net_pos)
        else:
            unrealized_pnl = (-curr_price+avg_price)*20*abs(net_pos)

        total_pnl = round(unrealized_pnl+realized_pnl,2)

        if total_pnl >= 0:
            return f"${total_pnl:,}"
        else:
            return f"-${abs(total_pnl):,}"
        
    @render.text
    @reactive.event(input.forward, input.buy, input.sell, input.reset, ignore_none=False)
    def compute_unrealized_pnl():
        net_pos = len(buy_price)-len(sell_price)
        realized_pnl = 0
        unrealized_pnl = 0

        if net_pos == 0:
            unrealized_pnl = 0
            return f"${unrealized_pnl}"
        
        curr_price = df["Close"][int(input.skip())+input.forward()+int(input.bars_to_show())]
        avg_price = np.average(buy_price[-abs(net_pos):]) if net_pos > 0 else np.average(sell_price[-abs(net_pos):])

        if net_pos > 0:
            unrealized_pnl = round((curr_price-avg_price)*20*abs(net_pos), 2)
        else:
            unrealized_pnl = round((-curr_price+avg_price)*20*abs(net_pos), 2)

        if unrealized_pnl >= 0:
            return f"${unrealized_pnl:,}"
        else:
            return f"-${abs(unrealized_pnl):,}"

app = App(app_ui, server)