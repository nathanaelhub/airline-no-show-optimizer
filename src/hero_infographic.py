import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm

class HeroInfographicGenerator:
    """
    Generate a powerful hero infographic for the airline-no-show-optimizer project.
    
    Creates a LinkedIn/portfolio-worthy visualization showing the transformation
    from empty seats to $10M revenue through ML-powered optimization.
    """
    
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.viz_dir = self.PROJECT_ROOT / 'results' / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Professional airline color palette
        self.colors = {
            'airline_blue': '#1E3A8A',      # Deep airline blue
            'airline_gold': '#F59E0B',      # Premium gold
            'success_green': '#10B981',     # Success green
            'light_blue': '#3B82F6',       # Light blue
            'light_gold': '#FCD34D',       # Light gold
            'text_dark': '#1F2937',        # Dark text
            'text_light': '#6B7280',       # Light text
            'background': '#F9FAFB',       # Light background
            'white': '#FFFFFF',            # Pure white
            'red': '#EF4444',              # Alert red
            'gray': '#9CA3AF'              # Neutral gray
        }
        
    def create_hero_infographic(self):
        """Create the main hero infographic."""
        
        # Create figure with exact dimensions for social media
        fig = plt.figure(figsize=(16, 8))  # 1200x600px equivalent
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create main axis
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        ax.axis('off')
        
        # Add title
        self.add_title(ax)
        
        # Create before/after sections
        self.create_before_section(ax)
        self.create_after_section(ax)
        
        # Add transformation arrow and metrics
        self.add_transformation_arrow(ax)
        
        # Add key innovations
        self.add_key_innovations(ax)
        
        # Add footer with branding
        self.add_footer(ax)
        
        # Save the infographic
        output_path = self.viz_dir / 'hero_infographic.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none',
                   pad_inches=0.5)
        
        print(f"✅ Hero infographic saved to: {output_path}")
        
        return fig
    
    def add_title(self, ax):
        """Add the main title."""
        ax.text(50, 55, 'From Empty Seats to $10M Revenue', 
               fontsize=32, fontweight='bold', ha='center', va='center',
               color=self.colors['text_dark'])
        
        ax.text(50, 51, 'ML-Powered Airline No-Show Optimization', 
               fontsize=18, ha='center', va='center',
               color=self.colors['text_light'], style='italic')
    
    def create_before_section(self, ax):
        """Create the 'Before' section with empty airplane."""
        # Before section background
        before_bg = FancyBboxPatch((2, 20), 38, 25, 
                                  boxstyle="round,pad=1", 
                                  facecolor=self.colors['white'],
                                  edgecolor=self.colors['gray'],
                                  linewidth=2)
        ax.add_patch(before_bg)
        
        # Before label
        ax.text(21, 42, 'BEFORE', fontsize=20, fontweight='bold', 
               ha='center', va='center', color=self.colors['red'])
        
        # Create airplane silhouette
        self.draw_airplane(ax, 21, 32, empty=True)
        
        # Add metrics
        ax.text(21, 26, '83%', fontsize=28, fontweight='bold', 
               ha='center', va='center', color=self.colors['red'])
        ax.text(21, 23, 'Load Factor', fontsize=14, ha='center', va='center',
               color=self.colors['text_dark'])
        
        # Lost revenue indicator
        ax.text(21, 21, '$0 Additional Revenue', fontsize=12, ha='center', va='center',
               color=self.colors['red'], weight='bold')
    
    def create_after_section(self, ax):
        """Create the 'After' section with full airplane."""
        # After section background
        after_bg = FancyBboxPatch((60, 20), 38, 25, 
                                 boxstyle="round,pad=1", 
                                 facecolor=self.colors['white'],
                                 edgecolor=self.colors['success_green'],
                                 linewidth=2)
        ax.add_patch(after_bg)
        
        # After label
        ax.text(79, 42, 'AFTER', fontsize=20, fontweight='bold', 
               ha='center', va='center', color=self.colors['success_green'])
        
        # Create full airplane silhouette
        self.draw_airplane(ax, 79, 32, empty=False)
        
        # Add metrics
        ax.text(79, 26, '95%', fontsize=28, fontweight='bold', 
               ha='center', va='center', color=self.colors['success_green'])
        ax.text(79, 23, 'Load Factor', fontsize=14, ha='center', va='center',
               color=self.colors['text_dark'])
        
        # Revenue indicator
        ax.text(79, 21, '$10.08M Annual Impact', fontsize=12, ha='center', va='center',
               color=self.colors['success_green'], weight='bold')
    
    def draw_airplane(self, ax, x, y, empty=True):
        """Draw airplane silhouette with seat indicators."""
        # Airplane body
        airplane_body = patches.Ellipse((x, y), 25, 6, 
                                       facecolor=self.colors['airline_blue'],
                                       alpha=0.8)
        ax.add_patch(airplane_body)
        
        # Wings
        wing_left = patches.Ellipse((x-5, y-1), 12, 3, 
                                   facecolor=self.colors['airline_blue'],
                                   alpha=0.8)
        wing_right = patches.Ellipse((x+5, y-1), 12, 3, 
                                    facecolor=self.colors['airline_blue'],
                                    alpha=0.8)
        ax.add_patch(wing_left)
        ax.add_patch(wing_right)
        
        # Tail
        tail = patches.Polygon([(x+10, y), (x+13, y+3), (x+13, y-3)], 
                              facecolor=self.colors['airline_blue'],
                              alpha=0.8)
        ax.add_patch(tail)
        
        # Seat indicators
        if empty:
            # Show empty seats (red circles)
            seat_color = self.colors['red']
            seat_positions = [(x-8, y), (x-4, y), (x, y), (x+4, y)]
            for seat_x, seat_y in seat_positions:
                if np.random.random() > 0.83:  # 83% occupancy
                    seat = Circle((seat_x, seat_y), 0.8, 
                                facecolor=seat_color, alpha=0.7)
                    ax.add_patch(seat)
        else:
            # Show full seats (green circles)
            seat_color = self.colors['success_green']
            seat_positions = [(x-8, y), (x-4, y), (x, y), (x+4, y), (x+8, y)]
            for seat_x, seat_y in seat_positions:
                if np.random.random() > 0.05:  # 95% occupancy
                    seat = Circle((seat_x, seat_y), 0.8, 
                                facecolor=seat_color, alpha=0.7)
                    ax.add_patch(seat)
    
    def add_transformation_arrow(self, ax):
        """Add transformation arrow with key metrics."""
        # Large arrow
        arrow = patches.FancyArrowPatch((42, 32), (58, 32),
                                       arrowstyle='->', 
                                       connectionstyle='arc3',
                                       mutation_scale=30,
                                       color=self.colors['airline_gold'],
                                       linewidth=4)
        ax.add_patch(arrow)
        
        # Metrics in the center
        metrics_bg = FancyBboxPatch((45, 28), 10, 8, 
                                   boxstyle="round,pad=0.5", 
                                   facecolor=self.colors['white'],
                                   edgecolor=self.colors['airline_gold'],
                                   linewidth=2)
        ax.add_patch(metrics_bg)
        
        ax.text(50, 34, 'ML IMPACT', fontsize=12, fontweight='bold', 
               ha='center', va='center', color=self.colors['airline_gold'])
        
        ax.text(50, 31.5, '+$27,622', fontsize=14, fontweight='bold', 
               ha='center', va='center', color=self.colors['success_green'])
        ax.text(50, 30, 'per flight', fontsize=10, ha='center', va='center',
               color=self.colors['text_dark'])
        
        ax.text(50, 28.5, '+12% seats', fontsize=12, fontweight='bold', 
               ha='center', va='center', color=self.colors['airline_blue'])
    
    def add_key_innovations(self, ax):
        """Add the three key innovations at the bottom."""
        # Background for innovations
        innovations_bg = FancyBboxPatch((5, 5), 90, 12, 
                                       boxstyle="round,pad=1", 
                                       facecolor=self.colors['white'],
                                       edgecolor=self.colors['airline_blue'],
                                       linewidth=2)
        ax.add_patch(innovations_bg)
        
        # Title
        ax.text(50, 15, 'KEY INNOVATIONS', fontsize=18, fontweight='bold', 
               ha='center', va='center', color=self.colors['airline_blue'])
        
        # Three innovations
        innovations = [
            {
                'title': 'Cost-Sensitive ML',
                'subtitle': '3:1 Asymmetric Penalties',
                'icon': '🎯',
                'x': 20
            },
            {
                'title': 'Monte Carlo Optimization',
                'subtitle': '10,000 Simulations',
                'icon': '🎲',
                'x': 50
            },
            {
                'title': '75 Domain Features',
                'subtitle': 'Passenger Psychology',
                'icon': '🧠',
                'x': 80
            }
        ]
        
        for innovation in innovations:
            # Icon background
            icon_bg = Circle((innovation['x'], 10), 2.5, 
                           facecolor=self.colors['light_blue'],
                           alpha=0.3)
            ax.add_patch(icon_bg)
            
            # Icon (using text as emoji)
            ax.text(innovation['x'], 10, innovation['icon'], 
                   fontsize=24, ha='center', va='center')
            
            # Title
            ax.text(innovation['x'], 7.5, innovation['title'], 
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   color=self.colors['text_dark'])
            
            # Subtitle
            ax.text(innovation['x'], 6.5, innovation['subtitle'], 
                   fontsize=10, ha='center', va='center',
                   color=self.colors['text_light'])
    
    def add_footer(self, ax):
        """Add footer with branding."""
        # Footer background
        footer_bg = Rectangle((0, 0), 100, 3, 
                             facecolor=self.colors['airline_blue'],
                             alpha=0.9)
        ax.add_patch(footer_bg)
        
        # Branding text
        ax.text(8, 1.5, 'Nathanael Johnson', 
               fontsize=14, fontweight='bold', ha='left', va='center',
               color=self.colors['white'])
        
        ax.text(50, 1.5, 'ML-Powered Revenue Optimization', 
               fontsize=14, ha='center', va='center',
               color=self.colors['airline_gold'], style='italic')
        
        ax.text(92, 1.5, 'airline-no-show-optimizer', 
               fontsize=12, ha='right', va='center',
               color=self.colors['white'])
    
    def create_linkedin_variant(self):
        """Create a LinkedIn-optimized variant."""
        # Create figure optimized for LinkedIn (1200x627px)
        fig = plt.figure(figsize=(16, 8.36))
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create main axis
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 62)
        ax.axis('off')
        
        # Add LinkedIn-optimized title
        ax.text(50, 57, 'From Empty Seats to $10M Revenue', 
               fontsize=30, fontweight='bold', ha='center', va='center',
               color=self.colors['text_dark'])
        
        ax.text(50, 53, 'Graduate Student • ML Engineer • Revenue Optimization Expert', 
               fontsize=16, ha='center', va='center',
               color=self.colors['text_light'], style='italic')
        
        # Add simplified content for LinkedIn
        # Results showcase
        results_bg = FancyBboxPatch((10, 25), 80, 25, 
                                   boxstyle="round,pad=2", 
                                   facecolor=self.colors['white'],
                                   edgecolor=self.colors['airline_blue'],
                                   linewidth=3)
        ax.add_patch(results_bg)
        
        # Key metrics in a grid
        metrics = [
            {'value': '$10.08M', 'label': 'Annual Impact', 'x': 25, 'y': 42},
            {'value': '95%', 'label': 'Load Factor', 'x': 50, 'y': 42},
            {'value': '500%', 'label': 'ROI', 'x': 75, 'y': 42},
            {'value': '129', 'label': 'ML Features', 'x': 25, 'y': 32},
            {'value': '0%', 'label': 'Denied Boarding', 'x': 50, 'y': 32},
            {'value': '2-3mo', 'label': 'Payback Period', 'x': 75, 'y': 32}
        ]
        
        for metric in metrics:
            ax.text(metric['x'], metric['y'], metric['value'], 
                   fontsize=24, fontweight='bold', ha='center', va='center',
                   color=self.colors['airline_blue'])
            ax.text(metric['x'], metric['y']-3, metric['label'], 
                   fontsize=12, ha='center', va='center',
                   color=self.colors['text_light'])
        
        # Add tagline
        ax.text(50, 20, 'Ready for Production • Built by a Graduate Student', 
               fontsize=18, fontweight='bold', ha='center', va='center',
               color=self.colors['success_green'])
        
        # Add footer
        self.add_footer(ax)
        
        # Save LinkedIn variant
        output_path = self.viz_dir / 'hero_infographic_linkedin.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none',
                   pad_inches=0.5)
        
        print(f"✅ LinkedIn hero infographic saved to: {output_path}")
        
        return fig


def main():
    """Generate hero infographic for the airline-no-show-optimizer project."""
    print("🎨 Generating Hero Infographic...")
    print("=" * 60)
    
    # Initialize infographic generator
    generator = HeroInfographicGenerator()
    
    # Create main hero infographic
    fig1 = generator.create_hero_infographic()
    
    # Create LinkedIn variant
    fig2 = generator.create_linkedin_variant()
    
    # Show summary
    print("\n🎯 HERO INFOGRAPHIC SUMMARY")
    print("=" * 60)
    print("✅ Main hero infographic created (1200x600px)")
    print("✅ LinkedIn variant created (1200x627px)")
    print("✅ Professional airline branding applied")
    print("✅ Portfolio-worthy design with key metrics")
    
    print(f"\n💡 KEY FEATURES:")
    print(f"   • Before/After transformation visualization")
    print(f"   • $10.08M revenue impact prominently displayed")
    print(f"   • Load factor improvement (83% → 95%)")
    print(f"   • Three key innovations highlighted")
    print(f"   • Professional airline blue/gold color scheme")
    print(f"   • Personal branding with name and tagline")
    
    print(f"\n📁 Files created:")
    print(f"   • {generator.viz_dir / 'hero_infographic.png'}")
    print(f"   • {generator.viz_dir / 'hero_infographic_linkedin.png'}")
    
    print(f"\n🚀 Ready for:")
    print(f"   • README.md hero image")
    print(f"   • LinkedIn post sharing")
    print(f"   • Portfolio presentations")
    print(f"   • Professional networking")
    
    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = main()
    plt.show()