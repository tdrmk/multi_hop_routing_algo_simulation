import pygame
import math
import itertools
import backend

pygame.init()

""" Just colors """
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
DARK_RED = (200, 0, 0)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 200)

""" Global GUI parameters """
WIDTH = 1600
HEIGHT = 800
FRAME_RATE = 60
CAPTION = 'Deterministic Model Routing Simulation'
ADVERSARY = 'adversary.png'
ALICE = 'alice.png'
BOB = 'bob.png'

""" Global GUI variables """
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(CAPTION)
clock = pygame.time.Clock()

""" Load all images """
adversary = pygame.image.load(ADVERSARY)
alice = pygame.image.load(ALICE)
bob = pygame.image.load(BOB)


def exit_function():
    pygame.quit()
    quit()


class Simulation(backend.NetworkGraph):
    """ Does probabilistic simulation """

    def __init__(self, num_nodes=12, num_paths=1, **kwargs):
        """ Model is used to generate curiosity and collaboration and also perform the routing algorithm"""
        self.model = backend.ProbabilisticModel
        display.fill(WHITE)
        """ Store the variable arguments for later use """
        self.kwargs = kwargs
        self.at_least_num_paths = num_paths
        backend.NetworkGraph.__init__(self, num_nodes)
        self.__reset_graph__()
        self.__update_model__()
        """ Set up colors for adversaries """
        self.__update_colors__()

        """ rectangle_list contains list of rectangles to update """
        self.rectangle_list = []

        """ event_handler is a dictionary which maps event type to list of event handlers """
        self.event_handlers = {}

        """ Set up the simulation using start_simulation method, currently set to an empty generator """
        self.simulation = iter(())
        """ Just set a default value, may not be used """
        self.obj_fn = backend.ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET

        """ Draw the buttons on initialization """
        self.draw_buttons()

    def __register_handler__(self, event_handler, *event_types):
        """ Register a event_handler for given set of events,
        event_handler takes event as input and return value is ignored """
        for event_type in event_types:
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append(event_handler)

    def __reset_graph__(self):
        """ ensure if possible 'num_paths', and update layout points """
        self.ensure_at_least_n_paths(self.at_least_num_paths)
        self.calculate_layout_points()

    def __update_model__(self):
        """ update curiosity and collaboration with new values """
        self.curiosity = self.model.random_curiosity(self.num_nodes, **self.kwargs)
        self.collaboration = self.model.random_collaboration(self.num_nodes, **self.kwargs)

    @staticmethod
    def create_colors(n):
        """ Number of possible values for individual channels """
        p = math.ceil(math.pow(n + 1, 1 / 3))
        """ Possible values for individual channels """
        c = [math.ceil((255 * i) / (p - 1)) for i in range(0, p)]

        def color_priority(color):
            """ Define a priority for colors """
            return sum(c if c > 0 else 512 for c in color)

        """ Sort color based on priority """
        return sorted(list(itertools.product(c, c, c)), key=color_priority, reverse=True)[:n]

    def __update_colors__(self):
        """ This function generates colors for adversarial nodes """
        """ Update the colors, a color for each adversary """
        self.colors = Simulation.create_colors(self.num_adversaries)

    def invalidate_rect(self, surf, rect):
        """ This function is used to draw an updated rect """
        display.blit(surf, rect)
        self.rectangle_list.append(rect)

    def main_loop(self):
        """ The main loop which handles events generated """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    """ exit main_loop on QUIT """
                    exit_function()

                if event.type in self.event_handlers:
                    """ If any event handler exists for a event type, the call it with event """
                    for event_handler in self.event_handlers[event.type]:
                        event_handler(event)

            """ Update the display at given rectangle list """
            if self.rectangle_list:
                pygame.display.update(self.rectangle_list)
                """ Reset the update list """
                self.rectangle_list = []
            """ Update next time after frame duration"""
            clock.tick(FRAME_RATE)

    def create_button(self, rect: pygame.Rect, text, color, hover_color, onclick_handler=None, button_scale=0.9):
        """ Returns an handler to which pass mouse events """

        def draw_button(__color):
            """ Draw a rectangle with given color on the game display,
             then draw the text at the center of the button, then add to update list """
            pygame.draw.rect(display, hover_color, rect)
            pygame.draw.rect(display, __color, Simulation.scale_rect(rect, button_scale))
            Simulation.draw_text(display, text, rect.height // 3, center=rect.center)

            self.rectangle_list.append(rect)

        """ Store the state of the button, that is whether mouse on top of button or not """
        hover = False
        draw_button(color)

        def event_handler(event):
            """ This function handles mouse events, it changes color on hover and
            calls onclick_handler on mouse-click"""
            nonlocal hover
            if event.type == pygame.MOUSEMOTION:
                if rect.collidepoint(event.pos):
                    if not hover:
                        """ Update the color """
                        hover = True
                        draw_button(hover_color)
                else:
                    if hover:
                        """ Update the color """
                        hover = False
                        draw_button(color)

            if event.type == pygame.MOUSEBUTTONDOWN:
                """ On-Click inside the button rect, execute callback if it exits """
                if rect.collidepoint(event.pos):
                    if onclick_handler:
                        onclick_handler()

        """ Register the event handler """
        self.__register_handler__(event_handler, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN)

    @staticmethod
    def create_color_image(image: pygame.Surface, target_color, src_color=WHITE, threshold=(10, 10, 10)):
        """ This function finds source color (color within threshold of src_color) in image
        and replaces it with target color and returns the new image """
        new_image = image.copy()
        pygame.transform.threshold(dest_surf=new_image, surf=image,
                                   search_color=src_color, threshold=threshold,
                                   set_color=target_color, inverse_set=True)
        return new_image

    @staticmethod
    def resize_image(image: pygame.Surface, width, height):
        """ This function returns a re-sized version of given image """
        return pygame.transform.scale(image, (int(width), int(height)))

    @staticmethod
    def create_adversary(target_color, width, height):
        """ Create an adversary image of given color """
        resized_image = Simulation.resize_image(adversary, width, height)
        return Simulation.create_color_image(resized_image, target_color)

    @staticmethod
    def create_source(width, height):
        """ Create a source image (alice) of given size """
        return Simulation.resize_image(alice, width, height)

    @staticmethod
    def create_destination(width, height):
        """ Create a destination image (bob) of given size """
        return Simulation.resize_image(bob, width, height)

    @staticmethod
    def draw_text(surf: pygame.Surface, text: str, font_size, center=None, position=None,
                  font_family='freesansbold.ttf',
                  text_color=BLACK):
        """ Draws a given text on a surface with given font size at given position on the surface with given color """
        """Create the font and render text with font with given color"""
        font = pygame.font.Font(font_family, int(font_size))
        text_surf = font.render(text, True, text_color)
        text_rect = text_surf.get_rect()
        if center:
            """ If center is specified then update the center of rect"""
            text_rect.center = (int(center[0]), int(center[1]))
        if position:
            """ If start position is specified, the update left and top """
            text_rect.left, text_rect.top = int(position[0]), int(position[1])
        surf.blit(text_surf, text_rect)

    @staticmethod
    def color_scale(value: float):
        """ This function maps a float value to a color. This function is used to visualize floats between 0 and 1"""
        assert 0 <= value <= 1.0
        color = (int(255 * value), int(255 - 255 * value), 0)
        return color

    @staticmethod
    def rect(left, top, width, height):
        """ Constructs a Rect using given positions which may possibly be float values """
        return pygame.Rect(int(left), int(top), int(width), int(height))

    @staticmethod
    def scale_rect(rectangle, scale=0.0, scale_x=0.0, scale_y=0.0):
        """ Utility function which scales the given rectangle, scale down if scale < 1, scale up if scale > 1"""
        if scale > 0:
            return rectangle.inflate(int(rectangle.width * (scale - 1)), int(rectangle.height * (scale - 1)))
        else:
            return rectangle.inflate(int(rectangle.width * (scale_x - 1)), int(rectangle.height * (scale_y - 1)))

    @staticmethod
    def scaled_rect(left, top, width, height, scale=0.0, scale_x=0.0, scale_y=0.0):
        """ Utility function which constructs a rectangle and scales it """
        return Simulation.scale_rect(Simulation.rect(left, top, width, height), scale, scale_x, scale_y)

    @staticmethod
    def move_rect(rectangle, x, y):
        """ Move the rectangle by given offset """
        return rectangle.move(int(x), int(y))

    """ These methods are used to draw GUI elements specific to this simulation """

    @staticmethod
    def construct_surface(left, top, width, height, return_dimensions=True):
        """ Construct a surface and fill it with a default color """
        rectangle = Simulation.rect(left, top, width, height)
        surface = pygame.Surface((rectangle.width, rectangle.height))
        surface.fill(WHITE)
        if return_dimensions:
            """ Return dimensions as well if requested """
            return surface, rectangle, rectangle.width, rectangle.height
        return surface, rectangle

    @staticmethod
    def draw_surface(dst_surface, src_surface, left, top):
        """ This function draws the src_surface into the dst_surface at given position """
        dst_surface.blit(src_surface, (int(left), int(top)))

    @staticmethod
    def draw_line(surface, color, start_x, start_y, end_x, end_y, width=1):
        """ A wrapper function around pygame.draw.line"""
        pygame.draw.line(surface, color, (int(start_x), int(start_y)), (int(end_x), int(end_y)), int(width))

    @staticmethod
    def draw_rect(surface, color, left, top, width, height, rect_width=0):
        pygame.draw.rect(surface, color, Simulation.rect(left, top, width, height), int(rect_width))

    @staticmethod
    def draw_ellipse(surface, color, left, top, width, height, ellipse_width=0):
        pygame.draw.ellipse(surface, color, Simulation.rect(left, top, width, height), ellipse_width)

    @staticmethod
    def color_rect_with_value(surface, rect, color_value):
        """ Fill the given rect on the surface with a given color_value according to color scale and
        also label the value for more readability """
        pygame.draw.rect(surface, Simulation.color_scale(color_value), rect)
        """Obtain font size such that color value fits in rect"""
        font_size = min(rect.height / 2, rect.width / 3)
        Simulation.draw_text(surface, '%0.2f' % color_value, font_size, rect.center, text_color=WHITE)

    def update_collaboration(self, collaboration, diagonal=1):
        assert diagonal == 0 or diagonal == 1
        """ This function updates the collaboration values on the GUI """
        surface, rectangle, width, height = Simulation.construct_surface(0, HEIGHT / 4, WIDTH / 4, HEIGHT / 2)
        num_adversaries = self.num_adversaries
        adversary_width, adversary_height = width / (num_adversaries + 1), height / (num_adversaries + 2)
        for i in range(0, num_adversaries):
            tmp_adversary = self.create_adversary(self.colors[i], adversary_width, adversary_height)
            Simulation.draw_surface(surface, tmp_adversary, 0, adversary_height * i)
        for j in range(0, num_adversaries):
            tmp_adversary = self.create_adversary(self.colors[j], adversary_width, adversary_height)
            Simulation.draw_surface(surface, tmp_adversary, adversary_width * (j + 1),
                                    adversary_height * num_adversaries)
        for i in range(0, num_adversaries):
            for j in range(0, i + diagonal):
                rect = Simulation.rect(adversary_width * (j + 1), adversary_height * i, adversary_width,
                                       adversary_height)
                """ Assuming adversaries in collaboration matrix start from 1, ie, 0 is start-point"""
                Simulation.color_rect_with_value(surface, rect, collaboration[i + 1, j + 1])

        """ Label the surface """
        self.draw_text(surface, 'COLLABORATION', adversary_height / 2,
                       (width / 2, height - adversary_height / 2))
        """ Draw the surface onto the display during the next update """
        self.invalidate_rect(surface, rectangle)

    def update_1d_metric(self, left, top, width, height, values, label):
        """ This function is used to update an 1D metric of adversaries on the GUI
            It has the format of metrics, adversaries, label
        """
        """ Note: values must includes source and end point values, which are ignored """
        assert len(values) == self.num_nodes
        surface, rectangle = Simulation.construct_surface(left, top, width, height, False)
        num_adversaries = self.num_adversaries
        adversary_width, adversary_height = width / num_adversaries, height / 3
        for i in range(num_adversaries):
            rect = Simulation.rect(adversary_width * i, 0, adversary_width, adversary_height)
            Simulation.color_rect_with_value(surface, rect, values[i + 1])
        """ Draw the adversaries """
        for i in range(num_adversaries):
            tmp_adversary = self.create_adversary(self.colors[i], adversary_width, adversary_height)
            Simulation.draw_surface(surface, tmp_adversary, adversary_width * i, adversary_height)
        Simulation.draw_text(surface, label, adversary_height / 2, (width / 2, height * 5 / 6))
        self.invalidate_rect(surface, rectangle)

    def update_curiosity(self, curiosity):
        self.update_1d_metric(0, HEIGHT * 3 / 4, WIDTH / 4, HEIGHT / 8, curiosity, 'CURIOSITY')

    def update_gathering_probability(self, gathering_probability):
        self.update_1d_metric(WIDTH * 3 / 4, HEIGHT * 4 / 8, WIDTH / 4, HEIGHT / 8, gathering_probability,
                              'GATHERING PROBABILITY')

    def update_decoding_probability(self, decoding_probability):
        self.update_1d_metric(WIDTH * 3 / 4, HEIGHT * 5 / 8, WIDTH / 4, HEIGHT / 8, decoding_probability,
                              'DECODING PROBABILITY')

    def update_breaking_probability(self, breaking_probability):
        self.update_1d_metric(WIDTH * 3 / 4, HEIGHT * 6 / 8, WIDTH / 4, HEIGHT / 8, breaking_probability,
                              'BREAKING PROBABILITY')

    def create_node(self, n, width, height):
        return Simulation.create_source(width, height) if n == self.start_node \
            else Simulation.create_destination(width, height) if n == self.end_node \
            else self.create_adversary(self.colors[n - 1], width, height)

    def update_path_choices(self, chosen_paths):
        surface, rectangle, width, height = Simulation.construct_surface(WIDTH * 3 / 4, 0, WIDTH / 4, HEIGHT / 2)
        paths, num_paths = self.reduced_paths, len(self.reduced_paths)
        """ Find the maximum length of the path """
        # num_lines = max(num_paths, self.num_nodes)
        max_path_length = max(map(len, paths))
        num_lines = max(num_paths, max_path_length, int(3 * self.num_nodes/ 4))
        node_width, node_height = width / (num_lines + 1), height / (num_lines + 1)

        for i in range(num_paths):
            """ Find number of times the paths[i] is chosen, the number of shares passing through i th path """
            num_times = chosen_paths.count(paths[i])
            font_size = min(node_width / 2, node_height * 3 / 4)
            Simulation.draw_text(surface, '%d' % num_times, font_size, (node_width / 2, node_height * (i + 1.5)))
            for j in range(len(paths[i])):
                n = paths[i][j]
                node = self.create_node(n, node_width, node_height)
                Simulation.draw_surface(surface, node, node_width * (j + 1), node_height * (i + 1))
        Simulation.draw_text(surface, 'PATHS', node_height, position=(node_width, 0))
        self.invalidate_rect(surface, rectangle)

    def draw_color_scale(self, resolution=1000):
        surface, rectangle, width, height = Simulation.construct_surface(0, HEIGHT * 7 / 8, WIDTH / 4, HEIGHT / 8)
        for i in range(resolution):
            color = Simulation.color_scale(i / resolution)
            x, line_width = width * i / resolution, width / resolution + 1
            Simulation.draw_line(surface, color, x, 0, x, height / 2, line_width)
        ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for tick in ticks:
            Simulation.draw_text(surface, '%0.1f' % tick, height / 8, (width * tick, height * 3 / 4))
        self.invalidate_rect(surface, rectangle)

    @staticmethod
    def locate_point(rect, norm_x, norm_y):
        """ This function takes a rectangle and the position of point with respect to the rectangle
        in normalized fashion, if norm_x and norm_y within (0, 1), then point returned lies rectangle
         returns the de-normalized point """
        point = int(rect.left + norm_x * rect.width), int(rect.top + norm_y * rect.height)
        print(rect, norm_x, norm_y, point)
        return point

    def update_num_shares(self, num_shares=0):
        """ Display the number of shares onto the GUI"""
        surface, rectangle, width, height = Simulation.construct_surface(0, 0, WIDTH / 4, HEIGHT / 8)
        font_size = min(width / 12, height)
        Simulation.draw_text(surface, "#SHARES: %d" % num_shares, font_size, center=(width / 2, height / 2))
        self.invalidate_rect(surface, rectangle)

    def update_status(self, message=""):
        """ Display some status message onto the GUI"""
        surface, rectangle, width, height = Simulation.construct_surface(0, HEIGHT / 8, WIDTH / 4, HEIGHT / 8)
        font_size = min(width / 12, height)

        Simulation.draw_text(surface, message, font_size, center=(width / 2, height / 2))
        self.invalidate_rect(surface, rectangle)

    def update_objective_value(self, value=0.0):
        """ Update the objective value on the GUI """
        surface, rectangle, width, height = Simulation.construct_surface(WIDTH * 3 / 4, HEIGHT * 7 / 8, WIDTH / 4,
                                                                         HEIGHT / 8)
        Simulation.draw_rect(surface, GRAY, width / 8, height / 16, width * 6 / 8, height * 6 / 16)
        position_x = width * (1 + 6 * value) / 8
        Simulation.draw_line(surface, BLACK, position_x, 0, position_x, height / 2)
        font_size = min(height / 3, width / 20)
        Simulation.draw_text(surface, 'OBJECTIVE: %0.6f' % value, font_size, center=(width / 2, height * 3 / 4))
        # surface.fill(BLACK)
        self.invalidate_rect(surface, rectangle)

    def draw_graph(self, chosen_paths):
        """ This function draws the network using the layout points,
        also draws the shares that each node directed obtained from the chosen paths """
        surface, rectangle, width, height = Simulation.construct_surface(WIDTH / 4, 0, WIDTH / 2, HEIGHT * 7 / 8)
        num_nodes = self.num_nodes
        node_width, node_height = rectangle.width / (num_nodes + 3), rectangle.height / (num_nodes + 3)
        edge_width = int(max(1, min(node_width, node_height) / 20))
        """ Obtain the position of adversarial nodes """
        node_rectangles = [Simulation.rect(
            node_width + node_width * num_nodes * self.nodes[i]['layout_point'].X,
            node_height + node_height * num_nodes * self.nodes[i]['layout_point'].Y,
            node_width, node_height
        ) for i in range(num_nodes)]

        """ Draw the links between the nodes """
        for (i, j) in self.edges():
            Simulation.draw_line(surface, BLUE, *node_rectangles[i].center, *node_rectangles[j].center, edge_width)

        """ Draw the nodes them-selves"""
        for i in range(num_nodes):
            """ Create the lines beneath the nodes, as nodes have alpha channel (transparency) """
            Simulation.draw_ellipse(surface, WHITE, *node_rectangles[i].topleft, *node_rectangles[i].size)
            Simulation.draw_surface(surface, self.create_node(i, node_width, node_height), *node_rectangles[i].topleft)

        if chosen_paths:
            """ If there are paths, then draw the shares """
            num_shares = len(chosen_paths)
            """ Get set of colors for the shares, make them distinct from node colors if possible """
            colors = Simulation.create_colors(num_shares + num_nodes)[num_nodes:]
            """ We plan to shares on both side of nodes, so keep the width less than half """
            share_width = node_width / 4
            """ Maximum nodes to draw on a side """
            limit = max(num_shares, num_nodes) / 2
            """ Calculate the share height """
            share_height = node_height / limit
            """ Keep track of number shares each node obtained, useful for positioning the shares """
            share_obtained = [0 for _ in range(num_nodes)]
            for i in range(num_shares):
                for j in chosen_paths[i]:
                    """ Check if we exceeded the limit for a given node, then draw on the other side"""
                    if share_obtained[j] < limit:
                        """ To draw on the left size """
                        share_left = node_rectangles[j].left
                        share_top = node_rectangles[j].bottom - share_height * (1 + share_obtained[j])
                    else:
                        """ to draw on the right size """
                        share_left = node_rectangles[j].right - share_width
                        share_top = node_rectangles[j].bottom - share_height * (1 + share_obtained[j] - limit)
                    """ Update the shares obtained of that node and draw the share"""
                    share_obtained[j] += 1
                    Simulation.draw_message_share(surface,
                                                  Simulation.rect(share_left, share_top, share_width, share_height),
                                                  colors[i], scale_message=0.7)
        """ Draw the graph onto the display """
        self.invalidate_rect(surface, rectangle)

    @staticmethod
    def scale_color(color, dim):
        """This utility function takes a color and fraction by which scales each of the color channel
        values, it makes sure that color is valid (lies within expected range)"""
        assert len(color) == 3 and dim >= 0

        def __scale__(__channel__):
            return int(max(0, min(255, dim * __channel__)))

        return __scale__(color[0]), __scale__(color[1]), __scale__(color[2])

    @staticmethod
    def draw_message_share(surface, rectangle, color, scale_color=0.9, scale_message=0.9):
        """ This function draws a share (one of the shares of the secret sharing) at a given location
        with given color, dim denotes the fraction by which color is scaled """
        assert 0 <= scale_message <= 1
        pygame.draw.rect(surface, WHITE, rectangle)
        rect = Simulation.scale_rect(rectangle, scale_message)
        scaled_color = Simulation.scale_color(color, scale_color)
        pygame.draw.polygon(surface, color, [rect.topleft, rect.topright, rect.bottomleft])
        pygame.draw.polygon(surface, scaled_color, [rect.bottomleft, rect.bottomright, rect.topright])

    def draw_buttons(self):
        self.create_button(Simulation.scaled_rect(WIDTH * 2 / 8, HEIGHT * 7 / 8, WIDTH / 8, HEIGHT / 8, 0.7), 'NEXT',
                           RED, DARK_RED, onclick_handler=self.simulation_next)
        self.create_button(Simulation.scaled_rect(WIDTH * 3 / 8, HEIGHT * 7 / 8, WIDTH / 8, HEIGHT / 8, 0.7), 'PLAY',
                           RED, DARK_RED, onclick_handler=self.simulation_play)
        self.create_button(Simulation.scaled_rect(WIDTH * 4 / 8, HEIGHT * 7 / 8, WIDTH / 8, HEIGHT / 8, 0.7), 'START',
                           RED, DARK_RED, onclick_handler=self.start_simulation)
        self.create_button(Simulation.scaled_rect(WIDTH * 5 / 8, HEIGHT * 7 / 8, WIDTH / 8, HEIGHT / 8, 0.7), 'RESET',
                           RED, DARK_RED, onclick_handler=self.reset_simulation)

    def start_simulation(self, obj_fn=backend.ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET):
        """ Set up the starting GUI and run simulation_next to start the simulation """
        self.update_curiosity(self.curiosity)
        self.update_collaboration(self.collaboration)
        self.draw_color_scale()
        self.obj_fn = obj_fn
        # TODO:
        self.simulation = backend.simulator(self.num_nodes, self.reduced_paths, self.curiosity, self.collaboration,
                                            obj_fn)
        print("START SIMULATION")
        self.simulation_next()

    def update_measures(self, num_shares, optimized, chosen_paths, gathering_probability, decoding_probability,
                        breaking_probability, objective_value):
        """ Update graph with chosen paths, update num shares, update probabilities and objective value """
        self.draw_graph(chosen_paths)
        self.update_num_shares(num_shares)
        self.update_path_choices(chosen_paths)
        self.update_gathering_probability(gathering_probability)
        self.update_decoding_probability(decoding_probability)
        self.update_breaking_probability(breaking_probability)
        self.update_objective_value(objective_value)
        if optimized:
            """ Set status if optimal """
            self.update_status("OPTIMAL")
        else:
            """ Just reset the status """
            self.update_status()

    def simulation_next(self):
        try:
            num_shares, optimized, chosen_paths, gathering_probability, decoding_probability, breaking_probability, \
            objective_value = next(self.simulation)
            self.update_measures(num_shares, optimized, chosen_paths, gathering_probability, decoding_probability,
                                 breaking_probability, objective_value)
        except StopIteration:
            """ This exception is throw if simulation is not yet started """
            # self.start_simulation()
            """ Raise an exception, an unexpected behavior """
            raise

    def simulation_play(self):
        """ Move the simulation to the next optimal solution """
        try:
            while True:
                """ Iterate till an optimal solution """
                num_shares, optimized, chosen_paths, gathering_probability, decoding_probability, \
                    breaking_probability, objective_value = next(self.simulation)
                if not optimized:
                    continue
                self.update_measures(num_shares, optimized, chosen_paths, gathering_probability, decoding_probability,
                                     breaking_probability, objective_value)
                """ Break after identifying optimal solution """
                break
        except StopIteration:
            """ This exception is throw if simulation is not yet started """
            # self.start_simulation()
            """ Raise an exception, an unexpected behavior """
            raise

    def reset_simulation(self):
        """ Reset the graph (generate a new graph) and update the model (generate new curiosity and collaborations) """
        self.__reset_graph__()
        self.__update_model__()
        """ And then start the simulation """
        self.start_simulation(self.obj_fn)


s = Simulation(12, 3)
s.start_simulation()
s.main_loop()
