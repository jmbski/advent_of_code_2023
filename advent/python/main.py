import json, re, os
from tqdm import tqdm


global_counter = {
    'count': 0
}

def count():
    global_counter['count'] += 1


def is_numeric(value) -> bool:
    """ Checks if a value is or can be cast as a number

    Args:
        value (Any): value to check

    Returns:
        bool: True if the value is a number, or is a string representation of one
    """

    if(isinstance(value, (int, float, complex))):
        return True
    elif(isinstance(value, str)):
        return value.isnumeric()
    return False
    
def combine_ints(*ints: list[int]) -> int:
    """ Combines any number of digits into a single new number in the sequential order they are
        provided in. 

    Args:
        ints (list | tuple): list of single digit numbers to combine

    Returns:
        int: the new combined number, 0 if there were no values passed in. If only 1 digit is passed, the 
            returned value is that digit in both the 10s and 1s place

    Example:
        combine_ints(1,4,3) # returns 143
        combine_ints() # returns 0
        combine_ints(5) # returns 55
    """

    combined = 0

    if(isinstance(ints,tuple)):
        ints = list(ints)
    
    if(isinstance(ints, list)):
        length = len(ints)

        if(length == 1):
            digit = ints[0]
            if(is_numeric(digit)):
                combined = (10 * digit) + digit
        elif(length >= 2):
            # reverse the values so that the power scaling applies correctly
            ints.reverse()

            scale = 1

            for digit in ints:
                combined += scale * digit

                # increase scale by power of 10 so the next digit is in the appropriate position
                scale *= 10
    return combined

def last(items: list[any]) -> any:
    """ Retrieves the last element in a list

    Args:
        items (list[any]): list to retrieve the value from

    Returns:
        any: the last item, or none if not a list or list is empty
    """

    if(isinstance(items, list)):
        length = len(items)
        if(length > 0):
            return items[length - 1]
        
def to_digit(value: str | int) -> int:
    """ Converts the string name of a digit to the int value

    Args:
        string (str): string to evaluate

    Returns:
        int: the int representation of the name, or None if not found
    """
    
    if(is_numeric(value)):
        return int(value)
    
    if(isinstance(value, str)):
        digit_names = [r'zero', r'one', r'two', r'three', r'four', r'five', r'six', r'seven', r'eight', r'nine']
        index = 0
        for name in digit_names:
            if(re.match(name, value)):
                return index
            index += 1
        

def get_line_digits(line: str) -> list[int]:
    """ Gets the first and last digit presented in the string using regex

    This also accounts for written out digits, so the substring 'nine' would read as 9

    Args:
        line (str): string of text to search

    Returns:
        list[int]:  the retrieved int values in the string, empty list if none were found
                    or an invalid value was passed in
    """    

    digits = []

    if(isinstance(line, str)):
        # regex: [] to group them individually, \d to match on numbers
        pattern = r'(?=([\d]|one|two|three|four|five|six|seven|eight|nine))'
        
        str_digits = re.findall(pattern, line)

        for digit in str_digits:
            digits.append(to_digit(digit))

    return [digits[0], last(digits)]

def retrieve_lines(path: str) -> list[str]:
    """ Reads a file at the given path and returns the lines from it

    Args:
        path (str): path to the file

    Returns:
        list[str]: the list of string lines from the file, or an empty list if the path is invalid
    """

    if(isinstance(path, str)):
        if(os.path.exists(path)):
            reader = open(path, 'r')
            return reader.readlines()
    return []
    
def sum_file_line_ints(path: str) -> int:
    """ Reads a file line by line, combines the digits on each and sums the number from each line

    Args:
        path (str): path to the data file

    Returns:
        int: sum total of all line number values
    """

    lines = retrieve_lines(path)
    
    line_nums = list(map(lambda x: combine_ints(*get_line_digits(x)), lines))

    total = 0
    for num in line_nums:
        if '0' in str(num):
            total += 1
    print(total)
    return sum(line_nums)

""" Day 2 """
class Cubes:
    red: int = 0
    green: int = 0
    blue: int = 0
    
    valid: bool = True
    id: int = -1

    def __init__(self, init: str, id: int) -> None:
        self.id = id
        if(isinstance(init, str)):
            init_split = init.split(',')
            for pull in init_split:
                pull_split = pull.split()
                count = int(pull_split[0])
                color = pull_split[1].strip()
                if(hasattr(self, color)):
                    setattr(self, color, count)

        self.check_valid()

    def check_valid(self, reds: int = 12, greens: int = 13, blues: int = 14 ):
        if(self.red > reds or self.green > greens or self.blue > blues):
            self.valid = False

class Game:
    cube_pulls: list[Cubes] = []
    id: int = -1
    valid: bool = True
    min_red: int = 0
    min_green: int = 0
    min_blue: int = 0
    power: int = 0

    def __init__(self, init: str) -> None:
        if(isinstance(init, str)):
            init_split = init.split(':')
            if(isinstance(init_split, list) and len (init_split) == 2):
                id_str = init_split[0]
                id_str = id_str.replace('Game ', '').strip()
                if(is_numeric(id_str)):
                    self.id = int(id_str)
                
                pulls_str = init_split[1]
                pulls = pulls_str.split(';')
                for pull in pulls:
                    cube_pull = Cubes(pull.strip(), self.id)
                    
                    if(cube_pull.valid == False):
                        self.valid = False
                    self.set_min_color(cube_pull)
                self.power = self.min_red * self.min_green * self.min_blue
                        
    def set_min_color(self, cubes: Cubes):
        colors = ['red','green', 'blue']
        for color in colors:
            min_color = 'min_' + color
            cube_count = getattr(cubes, color)
            self_min = getattr(self, min_color)

            if(self_min == 0):
                setattr(self, min_color, cube_count)
            else:
                setattr(self, min_color, max(self_min, cube_count))

def parse_day2_data(lines: list[str]):
    id_total = 0
    power_total = 0
    if(isinstance(lines, list)):
        for line in lines:
            game = Game(line)
            if(game.valid):
                id_total += game.id
            power_total += game.power
    return id_total, power_total

def day_2(path: str):
    if(isinstance(path, str) and os.path.exists(path)):
        lines = retrieve_lines(path)
        valid_sum, power_sum = parse_day2_data(lines)
        print(f'valid_sum: {valid_sum}, power_sum: {power_sum}')

""" Day 3 """
grid = []
def get_cell(x: int, y: int):
    if(y < len(grid) and y >= 0):
        row = grid[y]
        if(isinstance(row, list) and x < len(row) and x >= 0):
            return row[x]
        
def build_grid(lines: list[str]):
    
    for line in lines:
        row = []
        for char in line:
            if(char != '\n'):
                row.append(char)
        grid.append(row)

def print_adjacent(x, y, num):
    length = len(str(num))
    start_y = y - 1
    end_y = y + 2
    start_x = x - length - 1
    end_x = x + 1
    if(start_x < 0):
        start_x = 0
    if(start_y < 0):
        start_y = 0

    line = ''
    for row in grid[start_y:end_y]:
        for col in row[start_x:end_x]:
            line += col
        print(line)
        line = ''
    print('\n')
            
def get_numbers():
    numbers = []
    x = 0
    y = 0
    data = {
        'current_num': '',
        'num_adjacent': False,
        'num_gears': []
    }
    
    gear_map = {}

    def check_num():
        current_num = data.get('current_num')
        num_adjacent = data.get('num_adjacent')
        num_gears = data.get('num_gears')

        num = int(current_num)
        if(num_adjacent):
            numbers.append(num)
        if(len(num_gears) > 0):
            for gear in num_gears:
                key = f'{gear[0]},{gear[1]}'
                gear_value = gear_map.get(key)
                if(gear_value is None):
                    gear_map[key] = [num]
                else:
                    gear_map[key].append(num)

        data['current_num'] = ''
        data['num_gears'] = []
        data['num_adjacent'] = False

    for row in grid:
        for col in row:
            if(is_numeric(col)):
                data['current_num'] += col
                adjacent, gears = check_adjacent(x, y)

                if(not data['num_adjacent']):
                    data['num_adjacent'] = adjacent
                if(len(gears) > 0):
                    for gear in gears:
                        if(gear not in data.get('num_gears')):
                            data['num_gears'].append(gear)

            elif(data['current_num'] != ''):
                check_num()

            x += 1
        if(data['current_num'] != '' and data['num_adjacent']):
            check_num()
        y += 1
        x = 0

    gear_power = 0
    for nums in gear_map.values():
        if(len(nums) == 2):
            gear_power += nums[0] * nums[1]

    return numbers, gear_power

def check_adjacent(x: int, y: int):
    adjacent_cells = [
        [-1,-1], [0,-1], [1,-1],
        [-1, 0], [1, 0],
        [-1, 1], [0, 1], [1, 1]
    ]
    is_gear = False
    adjacent = False
    gears = []

    for cell in adjacent_cells:
        x2 = cell[0] + x
        y2 = cell[1] + y
        adj_cell = get_cell(x2, y2)
        if(adj_cell is not None and not is_numeric(adj_cell) and adj_cell != '.'):
            is_gear = adj_cell == '*'
            if(is_gear):
                gears.append([x2,y2])
            # need to account for multiples
    return adjacent, gears
    

def day_3():
    path = './data_files/day3_data.txt'
    lines = retrieve_lines(path)
    build_grid(lines)
    valid_parts, gear_power = get_numbers()
    pt1_answer = sum(valid_parts)
    
    print(f'pt1_answer: {pt1_answer}, gear_power: {gear_power}')

""" Day 4 """
def retrieve_day_lines(day: int = 4) -> list[str]:
    path = f'./data_files/day{day}_data.txt'
    reader = open(path, 'r')
    return reader.readlines()

def day_4():
    lines = retrieve_day_lines()
    total_points = 0
    for line in lines:
        winning_nums, your_nums, card_id = parse_cards(line)
        total_points += get_winning_score(winning_nums, your_nums)

    print(total_points)
    
def parse_cards(card: str):
    card_split = card.split(':')

    card_id = card_split[0]
    card_numbers = card_split[1].split('|')

    winning_nums_str = card_numbers[0].strip().split()
    your_nums_str = card_numbers[1].strip().split()

    winning_nums = []
    your_nums = []

    for num in winning_nums_str:
        if(is_numeric(num.strip())):
            winning_nums.append(int(num))
    for num in your_nums_str:
        if(is_numeric(num.strip())):
            your_nums.append(int(num))
    return winning_nums, your_nums, card_id

def get_winning_score(winning_nums, your_nums):
    points = 0
    for num in your_nums:
        if(num in winning_nums):
            if(points == 0):
                points = 1
            else:
                points *= 2
    
    return points
        
def day_4_pt2():
    lines = retrieve_day_lines()
    card_counts = []
    for line in lines:
        card_counts.append(1)
    index = 0
    for line in lines:

        count = card_counts[index] # 1

        matches = get_card_wins(line) # 4

        for i in range(1, 1+count):
            for j in range(1, 1+matches):
                card_counts[index + j] += 1

        index += 1

    print(sum(card_counts))
        

def get_card_wins(line: str) -> int:
    card_split = line.split(':')
    num_strs = card_split[1]
    num_split = num_strs.strip().split('|')

    winning_nums = num_split[0].strip().split()
    card_nums = num_split[1].strip().split()

    matches = 0

    for num in card_nums:
        if(num in winning_nums):
            matches += 1

    return matches

""" Day 5 """
def is_within(value, minimum, maximum, inclusive: bool = True) -> bool:
    if(inclusive):
        return value >= minimum and value <= maximum
    else:
        return value > minimum and value < maximum
    
class ValueRange:
    start: int = -1
    end: int = -1
    count: int = -1

    def __init__(self, start, count) -> None:
        self.start = start
        self.count = count
        self.end = start + count - 1

class ConversionSet:
    source_start: int = -1
    dest_start: int = -1
    count: int = 0
    source_max: int = -1
    dest_max: int = -1
    offset: int = -1

    def __init__(self, init_str) -> None:
        init_str_split = init_str.split()
        self.dest_start = int(init_str_split[0])
        self.source_start = int(init_str_split[1])
        self.count = int(init_str_split[2])
        self.source_max = self.source_start + self.count
        self.dest_max = self.dest_start + self.count
        self.offset = self.source_start - self.dest_start

class IORange:
    input_range: ValueRange = None
    output_range: ValueRange = None
    offset: int = -1

    def __init__(self, start, count, offset = 0) -> None:
        
        self.input_range = ValueRange(start, count)
        self.output_range = ValueRange(start - offset, count)
        self.offset = offset
        
    def check_num(num: int) -> bool:
        pass
    
    def print_self(self):
        print(f'range: {self.input_range.start}-{self.input_range.end} => {self.output_range.start}-{self.output_range.end}')

class ConversionMap:
    conversion_sets: list[ConversionSet] = None
    name: str = None
    ranges: list[IORange] = []

    def __init__(self, init_lines) -> None:
        self.name = init_lines[0].split()[0]
        self.conversion_sets = []
        self.ranges = []

        for set_data in init_lines[1:]:
            new_set = ConversionSet(set_data)
            self.conversion_sets.append(new_set)

    def calc_ranges(self, target_ranges: list[IORange] = []):
        self.ranges = []
        print(self.name)

        ranges = []
        for conv_set in self.conversion_sets:
            start = conv_set.source_start
            count = conv_set.count
            ranges.append(IORange(start, count, conv_set.offset))
        
        def get_gaps(ranges):
            input_sorted = sorted(ranges, key=lambda IO_instance: IO_instance.input_range.start)

            gap_start = 0
            gap_count = 0
            gaps = []
            for item in input_sorted:
                if(item.input_range.start > gap_start):
                    gap_count = item.input_range.start
                    gap_range = IORange(gap_start, gap_count)
                    gaps.append(gap_range)
                gap_start = item.input_range.end + 1
            return gaps

        gaps = get_gaps(ranges)

        if(gaps is not None):
            ranges.extend(gaps)
                
        self.ranges = sorted(ranges, key=lambda IO_instance: IO_instance.output_range.start)

        if(isinstance(target_ranges, list)):
            new_ranges = []
            output_sorted = self.ranges
            new_start = 0
            new_count = 0
            target_output = target_ranges[0]

            for io_range in output_sorted:
                for t_range in target_ranges:
                    gap_range = get_gap_range(t_range.output_range, io_range.input_range)
                    if(gap_range is not None):
                        new_io = IORange(gap_range.start, gap_range.count)
                        new_ranges.append(new_io)
                        
            new_gaps = get_gaps(new_ranges)
            print(len(new_gaps))

        
        """ for item in self.ranges:
            item.print_self() """

        return ranges
     
def get_gap_range(range_a: ValueRange, range_b: ValueRange):
    start = max(range_a.start, range_b.start)
    end = min(range_a.end, range_b.end)
    if( end >= start):
        dist = end - start + 1
        return ValueRange(start, dist)
    return None

def day_5_pt1():
    lines = retrieve_day_lines(5)
    map_lines = None
    maps = []
    seeds = []

    for line in lines:
        if(line.startswith('seeds:')):
            line_split = line.split(':')
            seed_strs = line_split[1].split()
            
            for i in range(0, len(seed_strs), 2):
                start = int(seed_strs[i])
                count = int(seed_strs[i+1])

                seeds.append(ValueRange(start, count))
                

        if(line.endswith(':\n')):
            if(isinstance(map_lines, list)):
                new_map = ConversionMap(map_lines)
                maps.append(new_map)
                new_map = None

            map_lines = []

            map_lines.append(line.replace('\n',''))
        elif(line.strip() != '' and isinstance(map_lines, list)):
            map_lines.append(line.replace('\n',''))

    if(isinstance(map_lines, list)):
        new_map = ConversionMap(map_lines)
        maps.append(new_map)
    
    get_ordered_range_map(maps)


def get_min_seed():
    pass

def map_seed_to_loc(maps: list[ConversionMap], seed: int):
    current_value = seed
    for mapping in maps:
        for conversion_set in mapping.conversion_sets:
            if(current_value >= conversion_set.source_start and current_value <= conversion_set.source_max):
                current_value -= conversion_set.offset
                break
    return current_value
    
def get_ordered_range_map(conversion_maps: list[ConversionMap]):
    range_maps = conversion_maps
    latest_input_map = None
    for r_map in range_maps:
        latest_input_map = r_map.calc_ranges(latest_input_map)

""" JSON Structure:
{
    name: "",
    range_maps: [
        {
            start: 0,
            end: 24,
            count: 25,
            offset: 5
        },
    ]
}
"""

def pretty_print(obj: dict):
    pretty = json.dumps(obj, indent=4)
    print(pretty)

def prop_append(obj: dict, key: str, value):
    if(isinstance(obj, dict) and isinstance(key, str)):
        if(isinstance(obj.get(key), list)):
            obj[key].append(value)
        else:
            obj[key] = [value]

def day5_pt2():
    lines = retrieve_day_lines(5)

    maps = []
    
    seed_ranges = []
    current_map = {}

    for line in lines:
        line = line.strip()
        if(line.startswith('seeds:')):
            seed_strs = line.split(':')[1].strip()
            seed_strs_split = seed_strs.split()
            for i in range(0, len(seed_strs_split), 2):
                start = int(seed_strs_split[i].strip())
                count = int(seed_strs_split[i+1].strip())
                end = start + count - 1

                seed_ranges.append({
                    'start': start,
                    'count': count,
                    'end': end,
                })

        else:
            if(':' in line):
                current_map['name'] = line.replace(':','')
            elif(line != ''):
                map_num_strs = line.split()
                map_nums = list(map(lambda x: int(x), map_num_strs))

                output_start, input_start, count = map_nums
                
                prop_append(current_map, 'ranges', build_range(output_start, input_start, count))

            if((line == '' or line == last(lines)) and current_map.get('name') is not None):
                maps.append(current_map)
                current_map = {}

    construct_full_map(maps)

def build_range(output_start, input_start, count) -> dict:

    offset = output_start - input_start
    if(count >= 0):
        return {
            'output_start': output_start,
            'output_end': output_start + count - 1,
            'input_start': input_start,
            'input_end': input_start + count - 1,
            'count': count, # -1 signifies all values above start
            'offset': offset
        }
    
    else:
        return {
            'output_start': output_start,
            'output_end': -1,
            'input_start': input_start,
            'input_end': -1,
            'count': count, # -1 signifies all values above start
            'offset': offset
        }

def get_input_gaps(ranges: list[dict]):
    by_input = sorted (ranges, key=lambda val_range: val_range.get('input_start'))
    cur_start = 0
    cur_end = 0
    cur_count = 0

    gap_ranges = []

    for val_range in by_input:
        start = val_range.get('input_start')

        if(cur_start < start):
            cur_end = start
            cur_count = cur_end - cur_start
            gap_ranges.append(build_range(cur_start, cur_start, cur_count))

        cur_start = val_range.get('input_end') + 1

    if(cur_start != 0):
        gap_ranges.append(build_range(cur_start, cur_start, -1))
    
    return gap_ranges

def get_range_input(v_range: dict, output: int):
    offset = v_range.get('offset')
    return output - offset

def get_range_output(v_range: dict, r_input: int):
    offset = v_range.get('offset')
    return r_input + offset

def get_overlap(range_a: dict, range_b: dict):
    """ Gets the overlap between the output of the previous (range_a) and the input of the next (range_b)
        If there is no overlap, returns None
    """
    # for output_end and input_end, -1 signifies all values above start
    a = range_a.get('output_start')
    b = range_a.get('output_end')
    x = range_b.get('input_start')
    y = range_b.get('input_end')

    """ 
        possible 'all values above' cases:
            b < 0 but y >= 0 = unbounded output with bounded input, so new range is bounded by y = build_range(output_start, input_end, )
                example = 54+ => 53-60, result should be a range of 53-60, so just return the input range
            b >= 0 and y < 0 = bounded output with unbounded input, so new range is bounded by b = build_range(output_start, input_start, )
                example = 50-97 => 54+, result should be a range of 54-97, in this case 54 = input_start, 97 = output_end, so new range is build_range(input_start, input_start, (output_end - input_start) + 1)
            b < 0 and y < 0 = unbounded output with unbounded input, just return output range, since values lower than output_start will be covered by the other two cases

    """

    # unbounded ends
    if(b < 0 and y >= 0 and a <= x):
        return range_b
    elif(b >= 0 and y < 0 and b >= x):
        input_start = get_range_input(range_a, x)
        return build_range(input_start, input_start, (b - x) + 1)
    elif(b < 0 and y < 0):
        return range_a
        
    c = max(a,x)
    d = min(b,y)

    if(d >= c):

        input_start = get_range_input(range_a, c)
        output_start = get_range_output(range_b, c)
        
        new_range = build_range(output_start, input_start, d - c + 1)
        return new_range
    
def remap_range(input_range, output_range):
    output_start = output_range.get('output_start')
    input_start = input_range.get('input_start')
    count = input_range.get('count')

    return build_range(output_start, input_start, count)

def map_outputs(prev_ranges: list[dict], new_ranges: list[dict]):
    by_output = sorted( prev_ranges, key=lambda x: x.get('output_start'))
    by_input = sorted( new_ranges, key=lambda x: x.get('input_start'))
    new_ranges = []
    for p_range in by_output:
        for n_range in by_input:
            overlap = get_overlap(p_range, n_range)
            if(overlap is not None):
                new_ranges.append(overlap)

    gaps = get_input_gaps(new_ranges)
    

    return new_ranges

def check_range_values(prev_start: int, prev_end: int, next_start: int, next_end: int):
    if((next_start <= next_end or next_end < 0) and (prev_start <= prev_end or prev_end < 0)):
        return True
    return False

def get_map_output(range_a: list[dict], range_b: list[dict]):
    prev_range = sorted(range_a, key=lambda x: x.get('output_start'))
    next_range = sorted(range_b, key=lambda x: x.get('input_start'))

    new_ranges = []

    p_range = prev_range[0]
    prev_start = p_range.get('output_start')
    prev_end = p_range.get('output_end')

    n_range = next_range[0]
    next_start = n_range.get('input_start')
    next_end = n_range.get('input_end')

    p_index = 0
    n_index = 0
    n_index_old = n_index
    p_index_old = p_index

    while( p_index < len(prev_range) and n_index < len(next_range)):

        if(p_index != p_index_old):
            p_range = prev_range[p_index]
            prev_start = p_range.get('output_start')
            prev_end = p_range.get('output_end')
            p_index_old = p_index

        cur_end = prev_end

        if(n_index != n_index_old):
            n_range = next_range[n_index]
            next_start = n_range.get('input_start')
            next_end = n_range.get('input_end')
            n_index_old = n_index

        while(n_index < len(next_range) and check_range_values(prev_start, prev_end, next_start, next_end)):
        
            if(is_within(prev_start, next_start, next_end) and prev_end >= 0 and next_end >= 0): # 0 within 0-14, true
                cur_end = min(cur_end, next_end) # 14
                # new range = output_start out(0) in(0) count(15)
                output_start = get_range_output(n_range, next_start)
                input_start = get_range_input(p_range, prev_start)
                new_ranges.append(build_range(output_start, input_start, cur_end - prev_start + 1))
                next_start = cur_end + 1
                prev_start = next_start 

            if(next_end < 0 and prev_end >= 0):
                input_start = get_range_input(p_range, prev_start)
                output_start = get_range_input(n_range, next_start)
                count = prev_end - prev_start + 1
                new_ranges.append(build_range(output_start, input_start, count))
                next_start = cur_end + 1
                prev_start = next_start

            if(prev_end < 0 and next_end >= 0):
                input_start = get_range_input(p_range, prev_start)
                output_start = get_range_output(n_range, next_start)
                count = next_end - next_start + 1
                new_ranges.append(build_range(output_start, input_start, count))
                next_start = cur_end + 1
                prev_start = next_start

            if(prev_end < 0 and next_end < 0):
                input_start = get_range_input(p_range, prev_start)
                output_start = get_range_output(n_range, next_start)
                new_ranges.append(build_range(output_start, input_start, -1))
                next_start = cur_end + 1
                prev_start = next_start

            if(next_start > next_end):
                n_index += 1

        if(prev_start > prev_end):
            p_index += 1
    
    if(last(prev_range) not in new_ranges):
        new_ranges.append(last(prev_range))

    return new_ranges


def construct_full_map(maps: list[dict]):
    full_map = [] # maybe dict instead? idk
    current_mapping = []
    ranges = []
    prev_ranges = None

    for val_map in maps:
        input_gaps = get_input_gaps(val_map.get('ranges'))
        ranges = val_map.get('ranges')
        ranges.extend(input_gaps)
        name = val_map.get('name')
        
        ranges.sort(key=lambda x: x.get('input_start'))
        print(f'\nMap: {name}')
        if(prev_ranges is not None):
            ranges = get_map_output(prev_ranges, ranges)
            
            #ranges.sort(key=lambda x: x.get('input_start'))
            """ pretty_print(ranges) """
            """ new_gaps = get_input_gaps(ranges) """
            # figure out how to remap


        prev_ranges = ranges

    ranges.sort(key=lambda x: x.get('output_start'))
    return ranges

def pause(output):
    print(output)
    resume = input()
    if(resume == 'x'):
        exit()

def day5_pt2_2():
    lines = retrieve_day_lines(5)

    all_converters: list[list[tuple]] = []

    seed_ranges = []

    name = ''

    converter_set: list[tuple] = []

    for line in lines:
        line = line.strip()
        if(line.startswith('seeds:')):
            seed_strs = line.split(':')[1].strip()
            seed_strs_split = seed_strs.split()
            for i in range(0, len(seed_strs_split), 2):
                start = int(seed_strs_split[i].strip())
                count = int(seed_strs_split[i+1].strip())
                end = start + count - 1

                seed_ranges.append([(start, end)])

        else:
            if(':' in line):
                name = line.replace(':','')
            elif(line != ''):
                map_num_strs = line.split()
                map_nums = list(map(lambda x: int(x), map_num_strs))

                output_start, input_start, count = map_nums
                input_end = input_start + count - 1

                offset = output_start - input_start

                converter_set.append((input_start, input_end, offset))
            elif(line == '' and len(converter_set) > 0):
                converter_set.sort(key=lambda x: x[0])
                all_converters.append(converter_set)
                converter_set = []

    converters_with_gaps = []
    current_start = 0

    for converter_set in all_converters:
        new_converter_set = []

        for converter in converter_set:
            start = converter[0]
            end = converter[1]
            
            if(start > current_start):
                new_converter_set.append((current_start, start - 1, 0))
                new_converter_set.append(converter)
                current_start = end + 1
            else:
                new_converter_set.append(converter)
                current_start = end + 1
        new_converter_set.append((current_start, -1, 0))
        current_start = 0
        new_converter_set.sort(key=lambda x: x[0])

        converters_with_gaps.append(new_converter_set)
    
    previous_converter_set = converters_with_gaps[0]
    previous_converter_set.sort(key=lambda x: x[0] + x[2])
    print(previous_converter_set)

    for converter_set in converters_with_gaps[1:]:
        new_converter_set = []
    
        origin_start = 0
        origin_end = 0
        origin_offset = 0
        org_output_start = 0
        org_output_end = 0
        next_start = 0
        next_end = 0
        next_offset = 0

        
        next_index = 0
        current_index = 0
        next_converter = converter_set[next_index]
        
        next_start, next_end, next_offset = next_converter

        for converter in previous_converter_set:
            origin_start, origin_end, origin_offset = converter
            org_output_start = origin_start + origin_offset
            org_output_end = origin_end + origin_offset

            new_end = origin_end

            # 
            #(0,49,0) => (0, 14, 39), (0,14,39) <=> (11,52,-11)
            # o_out_start = 39, n_in_start = 11, new = 39
            # o_out_end = 53, n_in_end = 52, new = 52
            # o_offset = 39, n_offset = -11, new = 28
            # o_out_start = new_end + 1, = 53
            # new = (0, 14, 28)
            # >> get next converter (53,60,-4)
            # o_out_start = 53, n_in_start 53, new = 53
            # o_out_end = 53, n_in_end = 60, new = 53
            # o_offset = 39, n_offset = -4, new = 35
            # new = (14, 14, 35)
            # >> get next original converter 

            while(next_index < len(converter_set)):
                if(org_output_start >= next_start and org_output_start <= next_end):
                    new_start = max(org_output_start, next_start) - origin_offset
                    new_end = min(org_output_end, next_end) - origin_offset
                    new_offset = origin_offset + next_offset

                    new_converter = (new_start, new_end, new_offset)
                    new_converter_set.append(new_converter)

                    next_start = new_end + 1 + origin_offset
                    org_output_start = next_start
                elif(next_end < 0 and org_output_end >= 0):
                    new_start = org_output_start - origin_offset
                    new_end = org_output_end - origin_offset
                    new_offset = origin_offset
                    new_converter = (new_start, new_end, new_offset)
                    org_output_start = org_output_end + 1
                    new_converter_set.append(new_converter)
                elif(next_end < 0 and org_output_end < 0):
                    new_start = org_output_start - origin_offset
                    new_end = -1
                    new_offset = 0

                    new_converter = (new_start, new_end, new_offset)
                    new_converter_set.append(new_converter)

                if(next_end < 0 or org_output_end < 0):
                    print(next_end, org_output_end)
                
                # handle unbounded ends
                if(next_end >= 0 and next_start > next_end):
                    next_index += 1
                    if(next_index < len(converter_set)):
                        next_converter = converter_set[next_index]
                        next_start, next_end, next_offset = next_converter
                if(org_output_start > org_output_end):

                    break
            

        new_converter_set.sort(key=lambda x: x[0] + x[2])
        previous_converter_set = new_converter_set

    print('final:\n')
    input_sort = sorted(previous_converter_set, key=lambda x: x[0])
    output_sort_asc = sorted(previous_converter_set, key=lambda x: x[0] + x[2])

    
    pretty_print(output_sort_asc)

    #pretty_print(all_converters)

def get_final_outputs(converters: list[list[tuple]]):
    final_outputs = []
    prev_converter_set = None
    current_input = 0

    for converter_set in converters:
        pass

"""
    95/100 = 2/x
    95x = 200
    x = 200/95
"""
day5_pt2_2()
