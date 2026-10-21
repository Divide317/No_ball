class CricketRuleEngine:
    def __init__(self):
        pass

    def get_powerplay_info(self, match_format: str, over: int, ball_index_in_innings: int):
        fmt = match_format.upper().strip()
        if fmt in ["T20", "T20I", "IPL"]:
            if over < 6:
                return {"phase": "Powerplay (Overs 1-6)", "max_outside_circle": 2, "max_bouncers": 2 if fmt == "IPL" else 1}
            return {"phase": "Middle/Death Overs (7-20)", "max_outside_circle": 5, "max_bouncers": 2 if fmt == "IPL" else 1}
        elif fmt == "ODI":
            if over < 10:
                return {"phase": "Powerplay 1 (Overs 1-10)", "max_outside_circle": 2, "max_bouncers": 2}
            elif 10 <= over < 40:
                return {"phase": "Powerplay 2 (Overs 11-40)", "max_outside_circle": 4, "max_bouncers": 2}
            else:
                return {"phase": "Powerplay 3 (Overs 41-50)", "max_outside_circle": 5, "max_bouncers": 2}
        elif fmt == "100-BALL":
            if ball_index_in_innings <= 25:
                return {"phase": "Powerplay (Balls 1-25)", "max_outside_circle": 2, "max_bouncers": 1}
            return {"phase": "Non-Powerplay (Balls 26-100)", "max_outside_circle": 5, "max_bouncers": 1}
        else:  # TEST
            return {"phase": "Test Match Playing Conditions", "max_outside_circle": 9, "max_bouncers": 2}

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
            reasons.append(f"Leg-side fielding limit exceeded: {fielders_leg_side} fielders on leg side (max 5 allowed, Law 28.4).")

        # 3. Behind-Square Leg-Side Fielders Limit - Law 28.4
        # There must be no more than 2 fielders on the leg side behind the popping crease.
        if fielders_behind_square > 2:
            is_no_ball = True
            reasons.append(f"Behind-square fielding limit exceeded: {fielders_behind_square} fielders behind square leg (max 2 allowed, Law 28.4).")

        # 4. Short-Pitched Deliveries (Bouncer) Limit - Law 41.6
        if is_current_ball_bouncer:
            total_bouncers = bouncers_in_over_before + 1
            if fmt in ["T20", "T20I"]:
                # Standard T20I allows max 1 bouncer per over. 2nd bouncer is a no-ball.
                if total_bouncers > 1:
                    is_no_ball = True
                    reasons.append(f"Bouncer limit exceeded for T20I: {total_bouncers} bouncers in this over (max 1 allowed, Law 41.6).")
            elif fmt == "IPL":
                # IPL playing conditions allow up to 2 bouncers per over. 3rd bouncer is a no-ball.
                if total_bouncers > 2:
                    is_no_ball = True
                    reasons.append(f"Bouncer limit exceeded for IPL: {total_bouncers} bouncers in this over (max 2 allowed, IPL Clause 41.6).")
            elif fmt in ["TEST", "ODI"]:
                # Test & ODI allow max 2 bouncers per over. 3rd bouncer is a no-ball.
                if total_bouncers > 2:
                    is_no_ball = True
                    reasons.append(f"Bouncer limit exceeded for {fmt}: {total_bouncers} bouncers in this over (max 2 allowed, Law 41.6).")
            elif fmt == "100-BALL":
                # In 100-ball (The Hundred), max 1 bouncer per 5-ball set.
                if total_bouncers > 1:
                    is_no_ball = True
                    reasons.append(f"Bouncer limit exceeded for 100-ball: {total_bouncers} bouncers in this set (max 1 per 5 balls allowed).")

        # 5. Fielding Circle Restrictions (Powerplay vs. Non-Powerplay)
        if fmt in ["T20", "T20I", "IPL"]:
            if over < 6:
                if fielders_outside_circle > 2:
                    is_no_ball = True
                    reasons.append(f"Powerplay fielding restriction violation: {fielders_outside_circle} fielders outside 30-yard circle (max 2 allowed in Overs 1-6).")
            else:
                if fielders_outside_circle > 5:
                    is_no_ball = True
                    reasons.append(f"Fielding restriction violation: {fielders_outside_circle} fielders outside 30-yard circle (max 5 allowed in Overs 7-20).")
        
        elif fmt == "ODI":
            if over < 10:
                if fielders_outside_circle > 2:
                    is_no_ball = True
                    reasons.append(f"ODI Powerplay 1 violation: {fielders_outside_circle} fielders outside the circle (max 2 allowed).")
            elif 10 <= over < 40:
                if fielders_outside_circle > 4:
                    is_no_ball = True
                    reasons.append(f"ODI Powerplay 2 violation: {fielders_outside_circle} fielders outside the circle (max 4 allowed).")
            else:
                if fielders_outside_circle > 5:
                    is_no_ball = True
                    reasons.append(f"ODI Powerplay 3 violation: {fielders_outside_circle} fielders outside the circle (max 5 allowed).")
                    
        elif fmt == "100-BALL":
            if ball_index_in_innings <= 25:
                if fielders_outside_circle > 2:
                    is_no_ball = True
                    reasons.append(f"100-ball Powerplay violation: {fielders_outside_circle} fielders outside the circle (max 2 allowed).")
            else:
                if fielders_outside_circle > 5:
                    is_no_ball = True
                    reasons.append(f"100-ball fielding restriction violation: {fielders_outside_circle} fielders outside the circle (max 5 allowed).")

        return is_no_ball, reasons

    def check_delivery_detailed(self,
                                match_format: str,
                                over: int,
                                ball_index_in_innings: int,
                                bouncers_in_over_before: int,
                                is_current_ball_bouncer: bool,
                                is_beamer: bool,
                                fielders_leg_side: int,
                                fielders_behind_square: int,
                                fielders_outside_circle: int,
                                overstep_prob: float = 0.0,
                                beamer_prob: float = 0.0):
        """
        Returns a rich structured breakdown of every ICC/MCC law check for SaaS telemetry.
        """
        fmt = match_format.upper().strip()
        pp_info = self.get_powerplay_info(fmt, over, ball_index_in_innings)
        total_bouncers = bouncers_in_over_before + (1 if is_current_ball_bouncer else 0)
        max_bouncers = pp_info["max_bouncers"]
        max_outside = pp_info["max_outside_circle"]

        checks = [
            {
                "id": "front_foot",
                "law": "MCC Law 21.5",
                "title": "Front-Foot Popping Crease",
                "passed": overstep_prob <= 0.50,
                "metric": f"{overstep_prob * 100:.1f}% overstep risk",
                "limit": "Some part of foot behind line",
                "source": "AI Vision (Crease CNN)",
            },
            {
                "id": "beamer",
                "law": "MCC Law 41.7",
                "title": "Waist-High Full Toss (Beamer)",
                "passed": not is_beamer,
                "metric": f"{beamer_prob * 100:.1f}% beamer risk" if beamer_prob > 0 else ("Flagged" if is_beamer else "Below waist"),
                "limit": "Below batter's waist on the full",
                "source": "AI Vision (Waist CNN) + Override",
            },
            {
                "id": "bouncer_quota",
                "law": "MCC Law 41.6",
                "title": "Short-Pitched Bouncer Quota",
                "passed": not (is_current_ball_bouncer and total_bouncers > max_bouncers),
                "metric": f"{total_bouncers} / {max_bouncers} in over",
                "limit": f"Max {max_bouncers} per over ({fmt})",
                "source": "Deterministic Rule Engine",
            },
            {
                "id": "powerplay_circle",
                "law": "ICC Clause 28.7",
                "title": f"30-Yard Circle ({pp_info['phase']})",
                "passed": True if fmt == "TEST" else (fielders_outside_circle <= max_outside),
                "metric": "No circle restriction" if fmt == "TEST" else f"{fielders_outside_circle} / {max_outside} outside",
                "limit": "Unrestricted (Test)" if fmt == "TEST" else f"Max {max_outside} outside circle",
                "source": "Deterministic Rule Engine",
            },
            {
                "id": "leg_side",
                "law": "MCC Law 28.4",
                "title": "On-Side (Leg-Side) Fielders",
                "passed": fielders_leg_side <= 5,
                "metric": f"{fielders_leg_side} / 5 on leg side",
                "limit": "Max 5 fielders on leg side",
                "source": "Deterministic Rule Engine",
            },
            {
                "id": "behind_square",
                "law": "MCC Law 28.4",
                "title": "Behind Popping Crease (Square Leg)",
                "passed": fielders_behind_square <= 2,
                "metric": f"{fielders_behind_square} / 2 behind square",
                "limit": "Max 2 behind square leg",
                "source": "Deterministic Rule Engine",
            },
        ]

        any_failed = any(not c["passed"] for c in checks)
        free_hit = any_failed and fmt in ["T20", "T20I", "IPL", "ODI", "100-BALL"]

        return {
            "powerplay_phase": pp_info["phase"],
            "max_outside_circle": max_outside,
            "max_bouncers_per_over": max_bouncers,
            "free_hit_awarded": free_hit,
            "checks": checks,
        }


if __name__ == "__main__":
    re = CricketRuleEngine()
    nb, reasons = re.check_delivery(
        match_format="T20",
        over=3,
        ball_index_in_innings=18,
        bouncers_in_over_before=0,
        is_current_ball_bouncer=False,
        is_beamer=False,
        fielders_leg_side=4,
        fielders_behind_square=1,
        fielders_outside_circle=3
    )
    print("Test 1 (T20 Powerplay over-fielding):", "NO-BALL!" if nb else "LEGAL", reasons)
