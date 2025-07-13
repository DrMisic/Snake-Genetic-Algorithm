import pygame
import random
import math
import matplotlib.pyplot as plt
from collections import deque
import time

# Inicijalizacija pygame
pygame.init()

# Konstante
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = GRID_WIDTH * GRID_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * GRID_SIZE + 100  # Dodatni prostor za info
FPS = 30

# Boje
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Pravci
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class DNA:
    def __init__(self, genes=None):
        # Genetski parametri za ponašanje zmije
        if genes is None:
            self.genes = {
                'food_attraction': random.uniform(0.5, 2.0),      # Koliko se privlači hrani
                'wall_avoidance': random.uniform(0.5, 2.0),      # Koliko izbegava zidove
                'self_avoidance': random.uniform(0.5, 2.0),      # Koliko izbegava sebe
                'forward_bias': random.uniform(0.1, 0.5),        # Tendencija da ide napred
                'risk_taking': random.uniform(0.1, 0.9),         # Koliko je spreman na rizik
                'exploration': random.uniform(0.1, 0.7),         # Koliko istražuje
                'corner_avoidance': random.uniform(0.3, 1.0),    # Izbegavanje uglova
                'path_planning': random.uniform(0.2, 0.8),       # Planiranje putanje
            }
        else:
            self.genes = genes.copy()
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        """Mutira gene sa određenom verovatnoćom"""
        for gene in self.genes:
            if random.random() < mutation_rate:
                # Dodaj gaussov šum
                change = random.gauss(0, mutation_strength)
                self.genes[gene] = max(0.1, min(2.0, self.genes[gene] + change))
    
    def crossover(self, other_dna):
        """Ukršta sa drugim DNA i vraća novo dete"""
        child_genes = {}
        for gene in self.genes:
            if random.random() < 0.5:
                child_genes[gene] = self.genes[gene]
            else:
                child_genes[gene] = other_dna.genes[gene]
        return DNA(child_genes)

class Snake:
    def __init__(self, x, y, dna=None):
        self.body = [(x, y)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.food = self.generate_food()
        self.score = 0
        self.alive = True
        self.steps = 0
        self.max_steps = 1000
        self.fitness = 0
        self.steps_without_food = 0
        self.dna = dna if dna else DNA()
        
        # Istorija pozicija za detektovanje petlji
        self.position_history = deque(maxlen=50)
        
    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if food not in self.body:
                return food
    
    def get_head(self):
        return self.body[0]
    
    def get_distance(self, pos1, pos2):
        """Manhattan udaljenost između dve pozicije"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_safe_position(self, pos):
        """Proverava da li je pozicija bezbedna"""
        x, y = pos
        
        # Provera granica
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return False
        
        # Provera kolizije sa telom
        if pos in self.body:
            return False
        
        return True
    
    def get_danger_score(self, pos):
        """Računa skor opasnosti za poziciju"""
        if not self.is_safe_position(pos):
            return 1000  # Maksimalna opasnost
        
        danger = 0
        x, y = pos
        
        # Opasnost od blizine zidova
        wall_distance = min(x, y, GRID_WIDTH-1-x, GRID_HEIGHT-1-y)
        if wall_distance <= 1:
            danger += 50 * self.dna.genes['wall_avoidance']
        
        # Opasnost od blizine tela
        for segment in self.body[1:]:  # Preskačemo glavu
            dist = self.get_distance(pos, segment)
            if dist <= 2:
                danger += (3 - dist) * 20 * self.dna.genes['self_avoidance']
        
        # Opasnost od zaglavljivanja u uglu
        adjacent_safe = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self.is_safe_position((x + dx, y + dy)):
                adjacent_safe += 1
        
        if adjacent_safe <= 1:
            danger += 30 * self.dna.genes['corner_avoidance']
        
        return danger
    
    def get_food_score(self, pos):
        """Računa skor privlačnosti hrane"""
        distance = self.get_distance(pos, self.food)
        if distance == 0:
            return 1000  # Maksimalna nagrada
        
        # Inverz udaljenosti * privlačnost hrane
        return (1.0 / distance) * 100 * self.dna.genes['food_attraction']
    
    def get_exploration_score(self, pos):
        """Računa skor za istraživanje"""
        # Favorizuje pozicije koje nisu nedavno posećene
        exploration_bonus = 0
        if pos not in self.position_history:
            exploration_bonus = 10 * self.dna.genes['exploration']
        
        return exploration_bonus
    
    def get_forward_bias_score(self, new_direction):
        """Daje bonus za nastavljanje u istom pravcu"""
        if new_direction == self.direction:
            return 15 * self.dna.genes['forward_bias']
        return 0
    
    def choose_direction(self):
        """Bira najbolji pravac na osnovu genetskih parametara"""
        head = self.get_head()
        possible_directions = [UP, DOWN, LEFT, RIGHT]
        
        # Uklanjamo suprotni pravac (ne može ići unazad)
        opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if opposite[self.direction] in possible_directions:
            possible_directions.remove(opposite[self.direction])
        
        best_direction = self.direction
        best_score = -float('inf')
        
        for direction in possible_directions:
            new_pos = (head[0] + direction[0], head[1] + direction[1])
            
            # Računanje ukupnog skora
            score = 0
            
            # Ako je pozicija nebezbedna, drastično smanji skor
            if not self.is_safe_position(new_pos):
                score -= 1000
            else:
                # Kombinuj sve faktore
                score += self.get_food_score(new_pos)
                score -= self.get_danger_score(new_pos)
                score += self.get_exploration_score(new_pos)
                score += self.get_forward_bias_score(direction)
                
                # Dodaj malo randomnosti za istraživanje
                score += random.uniform(-5, 5) * self.dna.genes['risk_taking']
            
            if score > best_score:
                best_score = score
                best_direction = direction
        
        return best_direction
    
    def move(self):
        if not self.alive:
            return
        
        # Izbor pravca
        self.direction = self.choose_direction()
        
        head = self.get_head()
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Provera kolizije sa zidovima
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.alive = False
            return
        
        # Provera kolizije sa telom
        if new_head in self.body:
            self.alive = False
            return
        
        # Dodaj novu poziciju u istoriju
        self.position_history.append(new_head)
        
        self.body.insert(0, new_head)
        
        # Provera da li je pojela hranu
        if new_head == self.food:
            self.score += 1
            self.food = self.generate_food()
            self.steps_without_food = 0
            self.max_steps += 100  # Produžava vreme kada pojede hranu
        else:
            self.body.pop()
            self.steps_without_food += 1
        
        self.steps += 1
        
        # Prekidanje ako je previše koraka bez hrane
        if self.steps_without_food > 200 or self.steps > self.max_steps:
            self.alive = False
    
    def calculate_fitness(self):
        """Računa fitnes na osnovu performansi"""
        # Osnovni fitnes na osnovu skora
        fitness = self.score * 1000
        
        # Bonus za preživljavanje
        fitness += self.steps * 2
        
        # Bonus za efikasnost (skor/vreme)
        if self.steps > 0:
            efficiency = self.score / (self.steps / 100.0)
            fitness += efficiency * 500
        
        # Kazna za dugotrajno lutanje
        if self.steps_without_food > 100:
            fitness -= (self.steps_without_food - 100) * 5
        
        # Bonus za duže zmije
        fitness += len(self.body) * 100
        
        self.fitness = max(0, fitness)
        return self.fitness
    
    def clone(self):
        """Kreira kopiju zmije"""
        clone = Snake(self.body[0][0], self.body[0][1])
        clone.dna = DNA(self.dna.genes)
        return clone

class GeneticAlgorithm:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_fitness = 0
        self.best_snake = None
        self.fitness_history = []
        self.average_fitness_history = []
        
        # Kreiranje početne populacije
        self.create_initial_population()
    
    def create_initial_population(self):
        """Kreira početnu populaciju"""
        self.population = []
        for _ in range(self.population_size):
            x = random.randint(2, GRID_WIDTH-3)
            y = random.randint(2, GRID_HEIGHT-3)
            snake = Snake(x, y)
            self.population.append(snake)
    
    def selection(self):
        """Turnir selekcija"""
        selected = []
        tournament_size = 5
        
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda s: s.fitness)
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Ukršta dva roditelja"""
        child_dna = parent1.dna.crossover(parent2.dna)
        child = Snake(random.randint(2, GRID_WIDTH-3), random.randint(2, GRID_HEIGHT-3), child_dna)
        return child
    
    def evolve(self):
        """Evolucija populacije"""
        # Računanje fitnesa
        for snake in self.population:
            snake.calculate_fitness()
        
        # Statistike
        fitnesses = [snake.fitness for snake in self.population]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        self.fitness_history.append(best_fitness)
        self.average_fitness_history.append(avg_fitness)
        
        # Najbolji zmija
        best_snake = max(self.population, key=lambda s: s.fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_snake = best_snake.clone()
        
        # Selekcija
        selected = self.selection()
        
        # Nova generacija
        new_population = []
        
        # Elitizam - čuvanje najboljih 20%
        elite_count = self.population_size // 5
        elite = sorted(self.population, key=lambda s: s.fitness, reverse=True)[:elite_count]
        for snake in elite:
            new_snake = snake.clone()
            new_snake.dna.mutate(mutation_rate=0.05, mutation_strength=0.05)  # Mala mutacija elite
            new_population.append(new_snake)
        
        # Ukrštanje i mutacija
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = self.crossover(parent1, parent2)
            child.dna.mutate(mutation_rate=0.15, mutation_strength=0.1)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Reset pozicija
        for snake in self.population:
            snake.body = [(random.randint(2, GRID_WIDTH-3), random.randint(2, GRID_HEIGHT-3))]
            snake.direction = random.choice([UP, DOWN, LEFT, RIGHT])
            snake.food = snake.generate_food()
            snake.score = 0
            snake.alive = True
            snake.steps = 0
            snake.steps_without_food = 0
            snake.position_history.clear()
    
    def run_generation(self):
        """Pokreće jedan korak simulacije"""
        alive_snakes = [snake for snake in self.population if snake.alive]
        
        # Pomeranje svih živih zmija
        for snake in alive_snakes:
            snake.move()
        
        # Vraća True ako još uvek ima živih zmija
        return len(alive_snakes) > 0
    
    def get_stats(self):
        """Vraća statistike trenutne populacije"""
        alive_count = len([s for s in self.population if s.alive])
        scores = [s.score for s in self.population]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        return {
            'generation': self.generation,
            'alive_count': alive_count,
            'avg_score': avg_score,
            'max_score': max_score,
            'best_fitness': self.best_fitness
        }

class GameVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("SnakeAI - Genetski algoritam")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
    
    def draw_snake(self, snake, color=GREEN):
        """Crta zmiju"""
        for i, segment in enumerate(snake.body):
            rect = pygame.Rect(segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            # Glava je svetlija
            if i == 0:
                pygame.draw.rect(self.screen, color, rect)
            else:
                # Telo je tamnije
                dark_color = tuple(max(0, c - 30) for c in color)
                pygame.draw.rect(self.screen, dark_color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 1)
    
    def draw_food(self, food):
        """Crta hranu"""
        rect = pygame.Rect(food[0] * GRID_SIZE, food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, RED, rect)
        pygame.draw.rect(self.screen, BLACK, rect, 2)
    
    def draw_info(self, stats):
        """Crta informacije o simulaciji"""
        info_y = WINDOW_HEIGHT - 100
        
        # Pozadina za info
        info_rect = pygame.Rect(0, info_y, WINDOW_WIDTH, 100)
        pygame.draw.rect(self.screen, GRAY, info_rect)
        
        # Tekst
        gen_text = self.font.render(f"Generacija: {stats['generation']}", True, WHITE)
        alive_text = self.font.render(f"Žive zmije: {stats['alive_count']}", True, WHITE)
        best_text = self.small_font.render(f"Najbolji fitnes: {stats['best_fitness']:.0f}", True, WHITE)
        score_text = self.small_font.render(f"Najbolji skor: {stats['max_score']}", True, WHITE)
        avg_text = self.small_font.render(f"Prosečan skor: {stats['avg_score']:.1f}", True, WHITE)
        
        self.screen.blit(gen_text, (10, info_y + 10))
        self.screen.blit(alive_text, (10, info_y + 40))
        self.screen.blit(best_text, (10, info_y + 70))
        self.screen.blit(score_text, (250, info_y + 10))
        self.screen.blit(avg_text, (250, info_y + 40))
    
    def draw_best_snake(self, snake):
        """Crta najbolju zmiju"""
        if snake and snake.alive:
            self.draw_food(snake.food)
            self.draw_snake(snake, BLUE)
    
    def draw_population(self, population, max_display=15):
        """Crta populaciju zmija"""
        alive_snakes = [snake for snake in population if snake.alive]
        
        if not alive_snakes:
            return
        
        # Crtanje hrane (sve zmije imaju istu hranu)
        self.draw_food(alive_snakes[0].food)
        
        # Crtanje zmija (ograničeno na max_display)
        for snake in alive_snakes[:max_display]:
            # Boja zavisi od skora
            if snake.score >= 5:
                color = BLUE  # Visok skor
            elif snake.score >= 2:
                color = YELLOW  # Srednji skor
            else:
                color = GREEN  # Nizak skor
            
            self.draw_snake(snake, color)
    
    def draw_instructions(self):
        """Crta instrukcije"""
        instructions = [
            "SPACE - brži/sporiji mod",
            "B - samo najbolja zmija",
            "R - restart simulacije",
            "ESC - izlaz"
        ]
        
        y_offset = 10
        for instruction in instructions:
            text = self.small_font.render(instruction, True, WHITE)
            self.screen.blit(text, (WINDOW_WIDTH - 220, y_offset))
            y_offset += 25

def main():
    """Glavna funkcija"""
    visualizer = GameVisualizer()
    ga = GeneticAlgorithm(population_size=30)
    
    running = True
    fast_mode = False
    show_best_only = False
    
    print("SnakeAI - Genetski algoritam")
    print("Kontrole:")
    print("SPACE - brži/sporiji mod")
    print("B - prikaži samo najbolju zmiju")
    print("R - restartuj simulaciju")
    print("ESC - izlaz")
    print("-" * 40)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    fast_mode = not fast_mode
                    print(f"Brzi mod: {'ON' if fast_mode else 'OFF'}")
                elif event.key == pygame.K_b:
                    show_best_only = not show_best_only
                    print(f"Prikaži samo najbolju: {'ON' if show_best_only else 'OFF'}")
                elif event.key == pygame.K_r:
                    ga = GeneticAlgorithm(population_size=30)
                    print("Simulacija restartovana!")
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Pokretanje generacije
        generation_active = ga.run_generation()
        
        if not generation_active:
            # Svi su mrtvi, nova generacija
            stats = ga.get_stats()
            print(f"Generacija {stats['generation']}: "
                  f"Najbolji skor = {stats['max_score']}, "
                  f"Prosečan skor = {stats['avg_score']:.1f}, "
                  f"Najbolji fitnes = {stats['best_fitness']:.0f}")
            
            ga.evolve()
        
        # Crtanje
        visualizer.screen.fill(BLACK)
        
        stats = ga.get_stats()
        
        if show_best_only and ga.best_snake:
            # Prikaži samo najbolju zmiju (kreiraj novu instancu)
            best_snake = ga.best_snake.clone()
            best_snake.body = [(GRID_WIDTH//2, GRID_HEIGHT//2)]
            best_snake.direction = UP
            best_snake.food = best_snake.generate_food()
            best_snake.alive = True
            best_snake.score = 0
            best_snake.steps = 0
            
            # Simuliraj najbolju zmiju
            step_count = 0
            while best_snake.alive and step_count < 1000:
                best_snake.move()
                step_count += 1
                
                if step_count % 50 == 0:  # Osvežava prikaz
                    visualizer.screen.fill(BLACK)
                    visualizer.draw_best_snake(best_snake)
                    visualizer.draw_info(stats)
                    visualizer.draw_instructions()
                    pygame.display.flip()
                    
                    if not fast_mode:
                        pygame.time.wait(50)
        else:
            # Prikaži populaciju
            visualizer.draw_population(ga.population)
        
        visualizer.draw_info(stats)
        visualizer.draw_instructions()
        
        pygame.display.flip()
        
        # Kontrola brzine
        if fast_mode:
            visualizer.clock.tick(FPS * 5)
        else:
            visualizer.clock.tick(FPS)
    
    pygame.quit()
    
    # Prikaz grafikona na kraju
    if ga.fitness_history:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(ga.fitness_history, 'b-', label='Najbolji fitnes')
        plt.plot(ga.average_fitness_history, 'r-', label='Prosečan fitnes')
        plt.title('Evolucija fitnesa kroz generacije')
        plt.xlabel('Generacija')
        plt.ylabel('Fitnes')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(ga.fitness_history, 'g-')
        plt.title('Najbolji fitnes (detaljan prikaz)')
        plt.xlabel('Generacija')
        plt.ylabel('Najbolji fitnes')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()