import base64
import io
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Circle, Rectangle, Arc, Polygon, Wedge, PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from plotly.figure_factory import create_distplot
import seaborn as sns
import numpy as np


def draw_court(ax=None, color='gray', lw=2, outer_lines=False):
    """Returns an axes with a basketball court drawn onto to it.
    This function draws a court based on the x and y-axis values that the NBA
    stats API provides for the shot chart data.  For example the center of the
    hoop is located at the (0,0) coordinate.  Twenty-two feet from the left of
    the center of the hoop in is represented by the (-220,0) coordinates.
    So one foot equals +/-10 units on the x and y-axis.
    Parameters
    ----------
    ax : Axes, optional
        The Axes object to plot the court onto.
    color : matplotlib color, optional
        The color of the court lines.
    lw : float, optional
        The linewidth the of the court lines.
    outer_lines : boolean, optional
        If `True` it draws the out of bound lines in same style as the rest of
        the court.
    Returns
    -------
    ax : Axes
        The Axes object with the court on it.
    """
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -12.5), 60, 0, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the right side 3pt lines, it's 14ft long before it arcs
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    # Create the right side 3pt lines, it's 14ft long before it arcs
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


def draw_restricted_area(ax, color='blue', alpha=0.3, lw=2):
    restricted_area = Wedge((0, 0), 40, theta1=0, theta2=180, linewidth=lw, color=color, alpha=alpha)
    ax.add_patch(restricted_area)

def draw_paint(ax, color='blue', alpha=0.3, lw=2):
    paint_vertices = [
        (-80, -47.5), (-80, 142.5), (80, 142.5), (80, -47.5),  
        (40, -47.5), (-40, -47.5), (-40, 0), (40, 0), (40, -47.5), (40, 0), (40,52.5), (-40,52.5), (-40, 0), (-40, -47.5)  
    ]
    paint_codes = [
        Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, 
        Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,  Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO  
    ]
    paint_path = Path(paint_vertices, paint_codes)
    paint_patch = PathPatch(paint_path, linewidth=lw, color=color, alpha=alpha, fill=True)
    ax.add_patch(paint_patch)

def draw_midrange(ax, color='blue', alpha=0.3, lw=2):
    midrange_vertices = [
        (220, 92.5), (220, -47.5), (80, -47.5), (80, 142.5), (-80, 142.5), (-80, -47.5), (-220, -47.5), (-220, 92.5), 
        (-138, 285), (138, 285), (220, 92.5), 
    ]
    midrange_codes = [
        Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.LINETO
    ]
    midrange_path = Path(midrange_vertices, midrange_codes)
    midrange_patch = PathPatch(midrange_path, linewidth=lw, color=color, alpha=alpha, fill=True)
    ax.add_patch(midrange_patch)

def draw_above_the_break_3(ax, color='blue', alpha=0.3, lw=2):
    above_the_break_3_vertices = [
         (220, 92.5), (250, 92.5), (250, 422.5), (-250, 422.5), (-250, 92.5), (-220, 92.5), (-138, 285), (138, 285), (220, 92.5), 
    ]
    above_the_break_3_codes = [
        Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.LINETO
    ]
    above_the_break_3_path = Path(above_the_break_3_vertices, above_the_break_3_codes)
    above_the_break_3_patch = PathPatch(above_the_break_3_path, linewidth=lw, color=color, alpha=alpha, fill=True)
    ax.add_patch(above_the_break_3_patch)

def draw_left_three_point_corner(ax, color='blue', alpha=0.3, lw=2):
    corner_three_a = Rectangle((-250, -47.5), 30, 140, linewidth=lw, color=color, alpha=alpha, fill=True)
    ax.add_patch(corner_three_a)

def draw_right_three_point_corner(ax, color='blue', alpha=0.3, lw=2):
    corner_three_b = Rectangle((250, -47.5), -30, 140, linewidth=lw, color=color, alpha=alpha, fill=True)
    ax.add_patch(corner_three_b)


def draw_shot_zones(ax=None, color='blue', alpha=0.3, lw=2):
    """
    Draws shot zones on a basketball court.
    
    Parameters
    ----------
    ax : Axes, optional
        The Axes object to plot the court onto.
    color : matplotlib color, optional
        The color of the shot zones.
    alpha : float, optional
        The transparency level of the shot zones.
    lw : float, optional
        The linewidth of the shot zone borders.
    
    Returns
    -------
    ax : Axes
        The Axes object with the shot zones drawn on it.
    """
    if ax is None:
        ax = plt.gca()

    # Draw each shot zone
    draw_restricted_area(ax, color=color, alpha=alpha, lw=lw)
    draw_paint(ax, color=color, alpha=alpha, lw=lw)
    draw_midrange(ax, color=color, alpha=alpha, lw=lw)
    draw_above_the_break_3(ax, color=color, alpha=alpha, lw=lw)
    draw_left_three_point_corner(ax, color=color, alpha=alpha, lw=lw)
    draw_right_three_point_corner(ax, color=color, alpha=alpha, lw=lw)

    return ax


def shot_chart(data,x, y, kind="scatter", title="", color="b", cmap=None,
               xlim=(-250, 250), ylim=(422.5, -47.5), draw_zones=False,
               court_color="gray", court_lw=2, outer_lines=False,
               flip_court=False, kde_shade=True, gridsize=None, ax=None,
               despine=False, **kwargs):
    """
    Returns an Axes object with player shots plotted.
    Parameters
    ----------
    x, y : strings or vector
        The x and y coordinates of the shots taken. They can be passed in as
        vectors (such as a pandas Series) or as columns from the pandas
        DataFrame passed into ``data``.
    data : DataFrame, optional
        DataFrame containing shots where ``x`` and ``y`` represent the
        shot location coordinates.
    kind : { "scatter", "kde", "hex" }, optional
        The kind of shot chart to create.
    title : str, optional
        The title for the plot.
    color : matplotlib color, optional
        Color used to plot the shots
    cmap : matplotlib Colormap object or name, optional
        Colormap for the range of data values. If one isn't provided, the
        colormap is derived from the valuue passed to ``color``. Used for KDE
        and Hexbin plots.
    {x, y}lim : two-tuples, optional
        The axis limits of the plot.
    court_color : matplotlib color, optional
        The color of the court lines.
    court_lw : float, optional
        The linewidth the of the court lines.
    outer_lines : boolean, optional
        If ``True`` the out of bound lines are drawn in as a matplotlib
        Rectangle.
    flip_court : boolean, optional
        If ``True`` orients the hoop towards the bottom of the plot.  Default
        is ``False``, which orients the court where the hoop is towards the top
        of the plot.
    kde_shade : boolean, optional
        Default is ``True``, which shades in the KDE contours.
    gridsize : int, optional
        Number of hexagons in the x-direction.  The default is calculated using
        the Freedman-Diaconis method.
    ax : Axes, optional
        The Axes object to plot the court onto.
    despine : boolean, optional
        If ``True``, removes the spines.
    kwargs : key, value pairs
        Keyword arguments for matplotlib Collection properties or seaborn plots.
    Returns
    -------
     ax : Axes
        The Axes object with the shot chart plotted on it.
    """

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = sns.light_palette(color, as_cmap=True)

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    ax.tick_params(labelbottom="off", labelleft="off")
    ax.set_title(title, fontsize=18)

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    if kind == "scatter":
        ax.scatter(data[x], data[y], c=color, **kwargs)

    elif kind == "kde":
        sns.kdeplot(data,x=x, y=y, fill=kde_shade, cmap=cmap, ax=ax,alpha=0.5, **kwargs)
        ax.set_xlabel('')
        ax.set_ylabel('')

#    elif kind == "kde":
#        ax = create_distplot(data,x=x, y=y,  **kwargs)

    elif kind == "hex":
        if gridsize is None:
            # Get the number of bins for hexbin using Freedman-Diaconis rule
            # This is idea was taken from seaborn, which got the calculation
            # from http://stats.stackexchange.com/questions/798/
            from seaborn.distributions import _freedman_diaconis_bins
            x_bin = _freedman_diaconis_bins(data[x])
            y_bin = _freedman_diaconis_bins(data[y])
            gridsize = int(np.mean([x_bin, y_bin]))
        
        hb = ax.hexbin(data[x].values, data[y].values, gridsize=gridsize, cmap=cmap, alpha=0.5, **kwargs)
        # add colorbar
        cb = plt.colorbar(hb)
        cb.set_label('Frequency')

    else:
        raise ValueError("kind must be 'scatter', 'kde', or 'hex'.")

    if draw_zones:
        draw_shot_zones(ax)

    # Set the spines to match the rest of court lines, makes outer_lines
    # somewhate unnecessary
    for spine in ax.spines:
        ax.spines[spine].set_lw(court_lw)
        ax.spines[spine].set_color(court_color)

    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    return ax

def shot_chart_zones(diff_dict, vmin, vmax, cbar_label="", title="", cmap="coolwarm", xlim=(-250, 250), ylim=(422.5, -47.5), flip_court=False,
                     court_color="gray", court_lw=2, ax=None, despine=False):
    """
    Returns an Axes object with shot zones plotted and colored by field goal
    percentage differences.

    Parameters
    ----------
    diff_dict : dict
        Dictionary mapping shot zone names to field goal percentage differences
        (e.g., { "restricted_area": -0.05, "midrange": 0.03, ... }).
    title : str, optional
        Title for the plot.
    cmap : matplotlib colormap, optional
        Colormap to use for field goal percentage differences.
    court_color : matplotlib color, optional
        Color of court lines.
    court_lw : float, optional
        Linewidth for court lines.
    ax : Axes, optional
        Matplotlib Axes object.
    despine : boolean, optional
        If True, removes the spines.
    Returns
    -------
    ax : Axes
        Matplotlib Axes object with zones colored.
    """
    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = sns.light_palette("blue", as_cmap=True)

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    ax.tick_params(labelbottom="off", labelleft="off")
    ax.set_title(title, fontsize=18)

    # Normalize FG% differences for the colormap
    norm = Normalize(vmin=vmin, vmax=vmax)  # Use vmin and vmax passed as arguments
    colormap = cm.get_cmap(cmap)

    # Draw the basketball court (assuming `draw_court` exists)
    draw_court(ax, color=court_color, lw=court_lw)

    # Helper function to add zones
    def add_colored_zone(ax, zone_function, color):
        zone_function(ax, color=color, alpha=0.8, lw=0)

    # Draw shot zones with colors
    for zone_name, diff in diff_dict.items():
        zone_color = colormap(norm(diff))
        if zone_name == "Restricted Area":
            add_colored_zone(ax, draw_restricted_area, zone_color)
        elif zone_name == "Mid-Range":
            add_colored_zone(ax, draw_midrange, zone_color)
        elif zone_name == "In The Paint (Non-RA)":
            add_colored_zone(ax, draw_paint, zone_color)
        elif zone_name == "Above the Break 3":
            add_colored_zone(ax, draw_above_the_break_3, zone_color)
        elif zone_name == "Left Corner 3":
            add_colored_zone(ax, draw_left_three_point_corner, zone_color)
        elif zone_name == "Right Corner 3":
            add_colored_zone(ax, draw_right_three_point_corner, zone_color)

    # Add a colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

    # Set the spines to match the rest of court lines, makes outer_lines
    # somewhate unnecessary
    for spine in ax.spines:
        ax.spines[spine].set_lw(court_lw)
        ax.spines[spine].set_color(court_color)

    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    return ax





def draw_plotly_court(fig, fig_width=600, margins=10):

    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200,
                    closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

    threept_break_y = 89.47765084
    three_line_col = "#777777"
    main_line_col = "#777777"

    fig.update_layout(
        # Line Horizontal
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=[
            dict(
                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ),
            dict(
                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ),

            dict(
                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                line=dict(color="#ec7607", width=1),
                fillcolor='#ec7607',
            ),
            dict(
                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ),
            dict(
                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color="#ec7607", width=1),
            ),

            dict(type="path",
                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),

            dict(
                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=250, y0=227.5, x1=220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=17.5, x1=80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=27.5, x1=80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=57.5, x1=80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=87.5, x1=80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),

            dict(type="path",
                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),

        ]
    )
    return True
def draw_restricted_area_plotly(fig, color='blue', alpha=0.3, lw=2):
    fig.add_shape(
        type="circle",
        x0=-40, y0=-40, x1=40, y1=40,  # Bounding box for the semicircle
        line=dict(color=color, width=lw),
        fillcolor=f"rgba({int(color == 'blue') * 0}, 0, {int(color == 'blue') * 255}, {alpha})",
        layer="below"
    )
    return fig

def convert_seaborn_to_img(fig):
    s = io.BytesIO()
    fig.savefig(s,format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getbuffer()).decode("ascii")
    return f'data:image/png;base64,{s}'

def shot_chart_plotly(data, x, y, kind="scatter", fig=None, name=None, **kwargs):
    if not fig:
        fig = go.Figure()
        draw_plotly_court(fig)

    if kind == "scatter":
        fig.add_trace(go.Scatter(x=data[x], y=data[y], mode='markers', name=name, **kwargs))
    elif kind == "kde":
        fig.add_trace(create_distplot([data[x], data[y]], group_labels=['made', 'missed']))

#    fig = draw_restricted_area_plotly(fig)
    return fig

"""def shot_chart_plotly(data, x, y, kind="scatter", title="", color="b", cmap=None,
               xlim=(-250, 250), ylim=(422.5, -47.5), draw_zones=False,
               court_color="gray", court_lw=2, outer_lines=False,
               flip_court=False, kde_shade=True, gridsize=None, **kwargs):
    
    Returns a Plotly figure with player shots plotted.
    Parameters
    ----------
    x, y : strings or vector
        The x and y coordinates of the shots taken. They can be passed in as
        vectors (such as a pandas Series) or as columns from the pandas
        DataFrame passed into ``data``.
    data : DataFrame, optional
        DataFrame containing shots where ``x`` and ``y`` represent the
        shot location coordinates.
    kind : { "scatter", "kde", "hex" }, optional
        The kind of shot chart to create.
    title : str, optional
        The title for the plot.
    color : matplotlib color, optional
        Color used to plot the shots
    cmap : matplotlib Colormap object or name, optional
        Colormap for the range of data values. If one isn't provided, the
        colormap is derived from the value passed to ``color``. Used for KDE
        and Hexbin plots.
    {x, y}lim : two-tuples, optional
        The axis limits of the plot.
    court_color : matplotlib color, optional
        The color of the court lines.
    court_lw : float, optional
        The linewidth the of the court lines.
    outer_lines : boolean, optional
        If ``True`` the out of bound lines are drawn in as a matplotlib
        Rectangle.
    flip_court : boolean, optional
        If ``True`` orients the hoop towards the bottom of the plot.  Default
        is ``False``, which orients the court where the hoop is towards the top
        of the plot.
    kde_shade : boolean, optional
        Default is ``True``, which shades in the KDE contours.
    gridsize : int, optional
        Number of hexagons in the x-direction.  The default is calculated using
        the Freedman-Diaconis method.
    kwargs : key, value pairs
        Keyword arguments for Plotly plots.
    Returns
    -------
    fig : Plotly figure
        The Plotly figure with the shot chart plotted on it.
    
    if cmap is None:
        cmap = 'Blues'

    if kind == "scatter":
        fig = px.scatter(data, x=x, y=y, color=color, title=title, **kwargs)
    elif kind == "kde":
        fig = px.density_contour(data, x=x, y=y, title=title, **kwargs)
        if kde_shade:
            fig = px.density_heatmap(data, x=x, y=y, title=title, **kwargs)
    elif kind == "hex":
        if gridsize is None:
            from seaborn.distributions import _freedman_diaconis_bins
            x_bin = _freedman_diaconis_bins(data[x])
            y_bin = _freedman_diaconis_bins(data[y])
            gridsize = int(np.mean([x_bin, y_bin]))
        fig = px.density_heatmap(data, x=x, y=y, nbinsx=gridsize, nbinsy=gridsize, color_continuous_scale=cmap, title=title, **kwargs)
    else:
        raise ValueError("kind must be 'scatter', 'kde', or 'hex'.")

    fig.update_xaxes(range=xlim)
    fig.update_yaxes(range=ylim)
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.update_layout(yaxis=dict(autorange="reversed"))

    draw_plotly_court(fig, color=court_color, lw=court_lw)

    return fig"""