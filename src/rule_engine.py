class CricketRuleEngine:
    def __init__(self):
        pass

    def check_delivery(self, 
                       match_format: str,
                       over: int,
                       ball_index_in_innings: int,
                       bouncers_in_over_before: int,
                       is_current_ball_bouncer: bool,
                       is_beamer: bool,
                       fielders_leg_side: int,
                       fielders_behind_square: int,
                       fielders_outside_circle: int):
        """
        Validates if a delivery violates any ICC/MCC laws of cricket.
        
        Returns:
            is_no_ball (bool): True if a rule was violated.
            reasons (list of str): List of specific rule violations.
        """
        reasons = []
        is_no_ball = False

        # Normalize format strings
        fmt = match_format.upper().strip()

        # 1. Waist-High Full Toss (Beamer) Check - Law 41.7
        if is_beamer:
            is_no_ball = True
            reasons.append("Waist-high full toss (beamer) on the full is illegal (Law 41.7).")

        # 2. Leg-Side Fielders Limit - Law 28.4
        # There must be no more than 5 fielders on the leg side at the instant of delivery.
        if fielders_leg_side > 5:
            is_no_ball = True
            reasons.append(f"Leg-side fielding limit exceeded: {fielders_leg_side} fielders on leg side (max 5 allowed).")

        # 3. Behind-Square Leg-Side Fielders Limit - Law 28.4
        # There must be no more than 2 fielders on the leg side behind the popping crease.
        if fielders_behind_square > 2:
            is_no_ball = True
            reasons.append(f"Behind-square fielding limit exceeded: {fielders_behind_square} fielders behind square leg (max 2 allowed).")

        # 4. Short-Pitched Deliveries (Bouncer) Limit - Law 41.6
        if is_current_ball_bouncer:
            total_bouncers = bouncers_in_over_before + 1
            if fmt == "T20":
                # Standard T20I allows max 1 bouncer per over. 2nd bouncer is a no-ball.
                # (Note: IPL allows 2 starting in 2024, but 1 remains the standard T20 International rule).
                if total_bouncers > 1:
                    is_no_ball = True
                    reasons.append(f"Bouncer limit exceeded for T20: {total_bouncers} bouncers in this over (max 1 allowed).")
            elif fmt in ["TEST", "ODI"]:
                # Test & ODI allow max 2 bouncers per over. 3rd bouncer is a no-ball.
                if total_bouncers > 2:
                    is_no_ball = True
                    reasons.append(f"Bouncer limit exceeded for Test/ODI: {total_bouncers} bouncers in this over (max 2 allowed).")
            elif fmt == "100-BALL":
                # In 100-ball (The Hundred), bowlers bowl sets of 5 or 10 balls.
                # The rule is max 1 bouncer per 5-ball set or 2 per 10-ball set.
                # If we assume 5-ball blocks (which are traditional over equivalents in The Hundred):
                if total_bouncers > 1:
                    is_no_ball = True
                    reasons.append(f"Bouncer limit exceeded for 100-ball: {total_bouncers} bouncers in this block (max 1 per 5 balls allowed).")

        # 5. Fielding Circle Restrictions (Powerplay vs. Non-Powerplay)
        # Powerplay rules depend on the format.
        if fmt == "T20":
            # Powerplay: Overs 0 to 5 (first 6 overs).
            if over < 6:
                if fielders_outside_circle > 2:
                    is_no_ball = True
                    reasons.append(f"Powerplay fielding restriction violation: {fielders_outside_circle} fielders outside the circle (max 2 allowed).")
            else:
                # Non-Powerplay: Max 5 fielders outside the circle.
                if fielders_outside_circle > 5:
                    is_no_ball = True
                    reasons.append(f"Fielding restriction violation: {fielders_outside_circle} fielders outside the circle (max 5 allowed).")
        
        elif fmt == "ODI":
            # ODI has 3 powerplay phases:
            # - Powerplay 1: Overs 0-9 (first 10 overs). Max 2 outside.
            # - Powerplay 2: Overs 10-39 (next 30 overs). Max 4 outside.
            # - Powerplay 3: Overs 40-49 (last 10 overs). Max 5 outside.
            if over < 10:
                if fielders_outside_circle > 2:
                    is_no_ball = True
                    reasons.append(f"ODI Powerplay 1 violation: {fielders_outside_circle} fielders outside the circle (max 2 allowed).")
            elif 10 <= over < 40:
                if fielders_outside_circle > 4:
                    is_no_ball = True
                    reasons.append(f"ODI Powerplay 2 violation: {fielders_outside_circle} fielders outside the circle (max 4 allowed).")
            else:  # over >= 40
                if fielders_outside_circle > 5:
                    is_no_ball = True
                    reasons.append(f"ODI Powerplay 3 violation: {fielders_outside_circle} fielders outside the circle (max 5 allowed).")
                    
        elif fmt == "100-BALL":
            # The Hundred: Powerplay is the first 25 balls of the innings.
            if ball_index_in_innings <= 25:
                if fielders_outside_circle > 2:
                    is_no_ball = True
                    reasons.append(f"100-ball Powerplay violation: {fielders_outside_circle} fielders outside the circle (max 2 allowed).")
            else:
                if fielders_outside_circle > 5:
                    is_no_ball = True
                    reasons.append(f"100-ball fielding restriction violation: {fielders_outside_circle} fielders outside the circle (max 5 allowed).")
                    
        # Note: Test matches do not have fielding circle restrictions.

        return is_no_ball, reasons

if __name__ == "__main__":
    # Quick sanity check
    re = CricketRuleEngine()
    
    # Test T20 Powerplay over-fielding
    nb, reasons = re.check_delivery(
        match_format="T20",
        over=3,
        ball_index_in_innings=18,
        bouncers_in_over_before=0,
        is_current_ball_bouncer=False,
        is_beamer=False,
        fielders_leg_side=4,
        fielders_behind_square=1,
        fielders_outside_circle=3  # Powerplay max is 2
    )
    print("Test 1 (T20 Powerplay over-fielding):", "NO-BALL!" if nb else "LEGAL", reasons)
    
    # Test T20 bouncer limit
    nb, reasons = re.check_delivery(
        match_format="T20",
        over=10,
        ball_index_in_innings=65,
        bouncers_in_over_before=1,
        is_current_ball_bouncer=True,  # 2nd bouncer
        is_beamer=False,
        fielders_leg_side=4,
        fielders_behind_square=1,
        fielders_outside_circle=4
    )
    print("Test 2 (T20 Bouncer limit):", "NO-BALL!" if nb else "LEGAL", reasons)
